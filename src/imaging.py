"""분류 입력을 위한 색상 제거 유틸.

색상(HSV)은 "손글씨가 어디 있는지 찾는" extraction 단계(preprocessing.ink_mask)에서만
쓴다. 어떤 펜을 들었는지는 작성자 고유의 안정적 특징이 아니라 그날그날 바뀌는 우연적
변수이므로, 분류기(임베딩 백본)에 들어가는 crop은 반드시 이 모듈을 거쳐 색상을 제거한다.
"""

import numpy as np
from PIL import Image

from src import config


def to_classification_input(img: Image.Image) -> Image.Image:
    """컬러 crop -> (선택적 이진화 포함) 그레이스케일 -> 3채널 복제.

    사전학습 백본은 대부분 3채널(RGB) 입력을 기대하므로, 그레이스케일/이진화 결과를
    채널 방향으로 복제해 RGB 형태로 되돌려 준다 (색 정보는 없고 밝기만 3채널에 동일).
    """
    gray = np.asarray(img.convert("L"))

    if config.BINARIZE_METHOD:
        thresh = _threshold(gray, config.BINARIZE_METHOD)
        # 배경(밝음)=255, 잉크(어두움)=0 극성을 유지한다.
        gray = np.where(gray > thresh, 255, 0).astype(np.uint8)

    return Image.merge("RGB", (Image.fromarray(gray),) * 3)


def _threshold(gray: np.ndarray, method: str):
    if method == "sauvola":
        from skimage.filters import threshold_sauvola
        return threshold_sauvola(gray, window_size=config.SAUVOLA_WINDOW, k=config.SAUVOLA_K)
    if method == "otsu":
        from skimage.filters import threshold_otsu
        return threshold_otsu(gray)
    raise ValueError(f"알 수 없는 이진화 방식: {method}")
