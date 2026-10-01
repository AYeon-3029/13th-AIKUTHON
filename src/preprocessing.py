"""① train PNG 페이지에서 손글씨 영역을 patch 단위로 추출한다.

test 는 대회 측이 이미 512x512 로 크롭해서 암호화 제공하므로 이 모듈은
train 에만 적용한다 (src/test_loader.py 로 별도 처리).

ink mask 는 두 가지를 OR 로 합친다:
  - 그레이스케일 마스크: 배경(종이)보다 충분히 어두운 픽셀 -> 검정 글씨
  - HSV 컬러 마스크: 채도가 높은 픽셀 -> 빨강/파랑 등 색 펜 주석
    (인쇄 텍스트/표/다이어그램은 대개 무채색이라 채도가 낮다)

색상은 "어디에 손글씨가 있는지 찾는" 이 단계에서만 쓴다. 실제로 저장되는 patch는
src.imaging.to_classification_input 을 거쳐 그레이스케일(+이진화)로 저장되므로,
분류기(임베딩 백본)는 색상을 전혀 보지 못한다 — 어떤 펜을 들었는지는 작성자의
안정적인 특징이 아니라 우연적 변수라서 과적합 shortcut이 될 수 있기 때문이다.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.imaging import to_classification_input

IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_pages(d: Path):
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)


def load_page(path: Path) -> Image.Image:
    """페이지 파일을 RGB PIL 이미지로 로딩. RGBA/팔레트는 흰 배경에 합성."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
    return img.convert("RGB")


def gray_ink_mask(arr: np.ndarray) -> np.ndarray:
    """배경 대비 충분히 어두운 픽셀 = 검정/무채색 잉크."""
    gray = np.asarray(Image.fromarray(arr).convert("L")).astype(np.int16)
    bg = float(np.median(gray))
    if bg < 128:  # 다크모드(어두운 배경 + 밝은 글씨)면 반전
        gray = 255 - gray
        bg = float(np.median(gray))
    return gray < max(config.GRAY_INK_MIN, bg - config.GRAY_INK_DELTA)


def color_ink_mask(arr: np.ndarray) -> np.ndarray:
    """채도가 높은 픽셀 = 색 펜 주석(빨강/파랑 등). 인쇄물은 대개 무채색."""
    hsv = np.asarray(Image.fromarray(arr).convert("HSV"))
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)
    return (sat >= config.HSV_SAT_THRESH) & (val >= config.HSV_VALUE_MIN)


def ink_mask(arr: np.ndarray) -> np.ndarray:
    return gray_ink_mask(arr) | color_ink_mask(arr)


def extract_patches(page_path: Path, out_dir: Path) -> int:
    """페이지 한 장 -> 잉크가 있는 patch 들을 out_dir 에 저장. 저장한 patch 수 반환."""
    img = load_page(page_path)

    w, h = img.size
    scale = config.PAGE_WIDTH / w
    img = img.resize((config.PAGE_WIDTH, max(1, round(h * scale))), Image.LANCZOS)

    arr = np.asarray(img)
    ink = ink_mask(arr)

    H, W = arr.shape[:2]
    if H < config.PATCH or W < config.PATCH:
        pad_h, pad_w = max(0, config.PATCH - H), max(0, config.PATCH - W)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
        ink = np.pad(ink, ((0, pad_h), (0, pad_w)), constant_values=False)
        H, W = arr.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for y in range(0, H - config.PATCH + 1, config.STRIDE):
        for x in range(0, W - config.PATCH + 1, config.STRIDE):
            if ink[y:y + config.PATCH, x:x + config.PATCH].mean() < config.MIN_INK_FRAC:
                continue
            tile = Image.fromarray(arr[y:y + config.PATCH, x:x + config.PATCH])
            # 색상(어떤 펜을 들었는지)은 여기 ink mask 를 만드는 데까지만 쓰고, 저장되는
            # patch 자체는 분류기 입력이므로 색상을 제거한다 (src/imaging.py).
            tile = to_classification_input(tile)
            tile.save(out_dir / f"{page_path.stem}_y{y:05d}_x{x:05d}.png")
            n += 1
    return n


def run(train_dir: Path, out_dir: Path, limit_pages: int | None = None) -> None:
    total_pages = total_patches = 0
    for w in config.WRITERS:
        pages = list_pages(train_dir / w)
        if limit_pages is not None:
            pages = pages[:limit_pages]
        for page in pages:
            page_out = out_dir / w / page.stem
            if page_out.is_dir():
                n = sum(1 for _ in page_out.glob("*.png"))
            else:
                n = extract_patches(page, page_out)
            total_pages += 1
            total_patches += n
    print(f"{total_pages}개 페이지 -> {total_patches}개 patch (cache: {out_dir})")


def main():
    ap = argparse.ArgumentParser(description="train 페이지에서 손글씨 patch 추출")
    ap.add_argument("--train-dir", type=Path, default=config.TRAIN_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.CACHE_DIR)
    ap.add_argument("--limit-pages", type=int, default=None,
                     help="작성자별 처리할 최대 페이지 수 (로컬 디버그용)")
    args = ap.parse_args()
    run(args.train_dir, args.out_dir, args.limit_pages)


if __name__ == "__main__":
    main()
