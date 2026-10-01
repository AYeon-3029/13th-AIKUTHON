"""경로, 클래스 순서, 하이퍼파라미터 상수를 한곳에 모아 둔다."""

import random
from pathlib import Path

import numpy as np

from src.test_loader import CLASSES

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train" / "train"     # <writer>/*.png
TEST_ROOT = ROOT                         # TestDS 가 {TEST_ROOT}/test/<id>.npz 를 찾음
SUBMISSION_TEMPLATE = ROOT / "sample_submission.csv"

CACHE_DIR = ROOT / "patches_cache"       # preprocessing.py 산출물
EMB_DIR = ROOT / "embeddings"            # embedding.py 산출물
MODEL_DIR = ROOT / "models"              # train.py 산출물
SUBMISSION_OUT = ROOT / "submission.csv"

WRITERS = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir()) if TRAIN_DIR.is_dir() else list(CLASSES)
assert WRITERS == CLASSES, (
    f"train/train 폴더명 순서({WRITERS})가 test_loader.CLASSES({CLASSES})와 다릅니다. "
    "라벨 인덱스가 어긋나면 채점이 전부 틀어지니 먼저 원인을 확인하세요."
)
W2I = {w: i for i, w in enumerate(WRITERS)}

SEED = 42

# ======================== 패치 추출 파라미터 (preprocessing.py) ========================
PAGE_WIDTH = 1280   # 페이지 가로폭 통일 기준
PATCH = 256          # 잘라내는 패치 한 변 크기
STRIDE = 192          # 슬라이딩 윈도우 간격 (25% 겹침)
MIN_INK_FRAC = 0.01  # 패치 내 잉크 비율이 이 값 미만이면 버림

# 그레이스케일 잉크: 배경보다 이 값 이상 어두우면 잉크로 간주
GRAY_INK_DELTA = 60
GRAY_INK_MIN = 40

# HSV 컬러 잉크: 채도(S)가 이 값 이상이면 색 잉크(빨강/파랑 등)로 간주.
# 인쇄 텍스트/표는 대개 무채색(검정)이라 채도가 낮으므로 색상 잉크와 구분된다.
HSV_SAT_THRESH = 60          # 0~255 스케일
HSV_VALUE_MIN = 30           # 너무 어두운(거의 검정) 픽셀은 그레이스케일 마스크가 이미 처리

# ======================== 분류 입력 색상 제거 (src/imaging.py) ========================
# 색상은 extraction(위 ink mask)에서만 쓰고, 분류기에 들어가는 crop은 항상 그레이스케일로
# 바꾼다. BINARIZE_METHOD 를 None 으로 두면 그레이스케일까지만 하고 이진화는 생략한다.
BINARIZE_METHOD = "sauvola"  # "sauvola" | "otsu" | None
SAUVOLA_WINDOW = 25           # 홀수여야 함
SAUVOLA_K = 0.2

# ======================== 임베딩 백본 (embedding.py) ========================
DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_MODEL = "dinov2_vits14"
DINOV2_IMG_SIZE = 224
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
