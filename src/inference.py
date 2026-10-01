"""④ 저장된 분류기로 test 를 추론해 submission.csv 를 만든다.

이 파일의 test 경로는 (test_loader.TestDS 로 복호화 -> transform -> 텐서 -> 임베딩
백본 forward -> 벡터) 만 다룬다. 복호화된 이미지를 화면에 띄우거나 파일로
저장하거나 사람이 보고 라벨을 붙이는 코드는 절대 추가하지 않는다 (대회 규칙, 실격 사유).
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from src import config
from src.embedding import embed_test


def load_or_compute_test_embeddings(backbone: str, device: str, batch_size: int = 64):
    path = config.EMB_DIR / f"{backbone}_test.npz"
    if path.is_file():
        data = np.load(path, allow_pickle=True)
        return data["vectors"], data["id"]
    path = embed_test(config.TEST_ROOT, config.SUBMISSION_TEMPLATE, backbone, device, batch_size)
    data = np.load(path, allow_pickle=True)
    return data["vectors"], data["id"]


def generate_submission(backbone: str, device: str, out_path, batch_size: int = 64):
    model_path = config.MODEL_DIR / f"{backbone}_logreg.joblib"
    if not model_path.is_file():
        raise RuntimeError(f"{model_path} 가 없습니다. train.py 를 먼저 실행하세요.")
    bundle = joblib.load(model_path)
    clf = bundle["model"]

    vectors, ids = load_or_compute_test_embeddings(backbone, device, batch_size)
    preds = clf.predict(vectors)

    sub = pd.DataFrame({"id": ids, "label": preds.astype(int)})

    template = pd.read_csv(config.SUBMISSION_TEMPLATE)
    sub = template[["id"]].merge(sub, on="id", how="left")
    if sub["label"].isna().any():
        missing = sub[sub["label"].isna()]["id"].tolist()
        raise RuntimeError(f"임베딩이 없는 id 가 있습니다: {missing[:5]}...")
    sub["label"] = sub["label"].astype(int)

    sub.to_csv(out_path, index=False)
    print(f"submission 저장: {out_path} ({len(sub)}행)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="submission.csv 생성")
    ap.add_argument("--backbone", choices=["dinov2", "clip"], default="dinov2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=str, default=str(config.SUBMISSION_OUT))
    args = ap.parse_args()

    config.set_seed()
    generate_submission(args.backbone, args.device, args.out, args.batch_size)


if __name__ == "__main__":
    main()
