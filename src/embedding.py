"""② frozen 사전학습 백본(DINOv2 / CLIP)으로 임베딩을 추출한다.

train: patches_cache/<writer>/<page>/*.png 를 순회해 patch-level 임베딩을 저장.
      (patch 는 preprocessing.py 가 이미 그레이스케일/이진화해 저장했으므로 그대로 사용)
test : src.test_loader.TestDS 로 복호화한 컬러 이미지를 raw_test_transform() 으로
       그레이스케일/이진화한 뒤 텐서로만 다뤄 임베딩을 저장한다.
       (대회 규칙: 복호화 이미지를 화면에 띄우거나 파일로 저장하는 코드는 절대 넣지 않는다.
        아래 test 경로는 텐서 -> 모델 forward -> 벡터 저장 만 수행한다.)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config
from src.imaging import to_classification_input
from src.test_loader import TestDS


class PatchListDataset(Dataset):
    """(파일 경로, writer, page_id) 리스트를 로딩하는 단순 Dataset."""

    def __init__(self, files, writers, page_ids, transform):
        self.files, self.writers, self.page_ids, self.transform = files, writers, page_ids, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), self.writers[i], self.page_ids[i]


def _imagenet_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_backbone(name: str, device: str):
    """frozen 사전학습 백본과 그에 맞는 (resize/normalize) transform 을 반환한다.

    반환되는 transform 은 색상 제거를 포함하지 않는다 — train patch 는 이미
    preprocessing.py 에서 그레이스케일(+이진화)로 저장돼 있으므로 그대로 쓰면 되고,
    test 의 원본 컬러 이미지에는 raw_test_transform() 으로 감싸서 써야 한다.
    """
    if name == "dinov2":
        model = torch.hub.load(config.DINOV2_HUB_REPO, config.DINOV2_HUB_MODEL)
        transform = _imagenet_transform(config.DINOV2_IMG_SIZE)
    elif name == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
        )
        transform = preprocess
    else:
        raise ValueError(f"알 수 없는 backbone: {name}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, transform


def raw_test_transform(backbone_transform):
    """test_loader 가 돌려주는 원본 컬러 이미지 -> 색상 제거 -> 백본 transform.

    복호화된 test 이미지는 컬러 상태로 들어오므로, 여기서 색상을 제거한 뒤에만
    백본에 넣는다 (extraction 단계와 동일하게, 분류기는 색상을 보지 않는다).
    """
    return transforms.Compose([
        transforms.Lambda(to_classification_input),
        backbone_transform,
    ])


@torch.no_grad()
def _encode(model, backbone: str, x: torch.Tensor) -> torch.Tensor:
    if backbone == "clip":
        return model.encode_image(x)
    return model(x)  # DINOv2 forward() 는 이미 CLS 토큰 임베딩을 반환


def embed_train(cache_dir: Path, backbone: str, device: str, batch_size: int = 64,
                 limit: int | None = None) -> Path:
    model, transform = load_backbone(backbone, device)

    files, writers, page_ids = [], [], []
    for w in config.WRITERS:
        wdir = cache_dir / w
        if not wdir.is_dir():
            continue
        for page_dir in sorted(p for p in wdir.iterdir() if p.is_dir()):
            for f in sorted(page_dir.glob("*.png")):
                files.append(f)
                writers.append(w)
                page_ids.append(f"{w}/{page_dir.name}")
    if limit is not None:
        files, writers, page_ids = files[:limit], writers[:limit], page_ids[:limit]
    if not files:
        raise RuntimeError(f"{cache_dir} 에 patch 가 없습니다. preprocessing.py 를 먼저 실행하세요.")

    ds = PatchListDataset(files, writers, page_ids, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    vecs = []
    for x, _, _ in loader:
        x = x.to(device)
        vecs.append(_encode(model, backbone, x).cpu().numpy())
    vecs = np.concatenate(vecs, axis=0)

    out_path = config.EMB_DIR / f"{backbone}_train.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        vectors=vecs,
        writer_label=np.array([config.W2I[w] for w in writers], dtype=np.int64),
        writer=np.array(writers),
        page_id=np.array(page_ids),
        file=np.array([str(f) for f in files]),
    )
    print(f"train 임베딩 {vecs.shape} 저장: {out_path}")
    return out_path


def embed_test(root: Path, submission_csv: Path, backbone: str, device: str,
                batch_size: int = 64, limit: int | None = None) -> Path:
    model, transform = load_backbone(backbone, device)
    transform = raw_test_transform(transform)

    ids = pd.read_csv(submission_csv)["id"].tolist()
    if limit is not None:
        ids = ids[:limit]

    ds = TestDS(ids, str(root), transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    vecs, out_ids = [], []
    for x, batch_ids in loader:
        x = x.to(device)
        vecs.append(_encode(model, backbone, x).cpu().numpy())
        out_ids.extend(batch_ids)
    vecs = np.concatenate(vecs, axis=0)
    print(f"test 임베딩 shape={vecs.shape} dtype={vecs.dtype}")  # 벡터 통계만 로깅, 이미지 자체는 다루지 않음

    out_path = config.EMB_DIR / f"{backbone}_test.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, vectors=vecs, id=np.array(out_ids))
    print(f"test 임베딩 저장: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="frozen 백본으로 임베딩 추출")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--backbone", choices=["dinov2", "clip"], default="dinov2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="로컬 디버그용 샘플 수 제한")
    args = ap.parse_args()

    config.set_seed()
    if args.split == "train":
        embed_train(config.CACHE_DIR, args.backbone, args.device, args.batch_size, args.limit)
    else:
        embed_test(config.TEST_ROOT, config.SUBMISSION_TEMPLATE, args.backbone,
                   args.device, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
