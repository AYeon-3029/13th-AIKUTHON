# AIKU 필적 분류 (Writer Identification)

강의 슬라이드 위에 덧씌워진 손글씨 주석을 보고, 10명의 작성자 중 누가 썼는지 분류하는 파이프라인.

## 폴더 구조

```
12-aikuthon/
├── train/train/<writer>/*.png   # 학습 이미지 829장 (원본 해상도 페이지)
├── test/<id>.npz                # 암호화된 평가 데이터 490개 (512×512, XOR)
├── sample_submission.csv
├── src/
│   ├── config.py                # 경로/클래스 순서/하이퍼파라미터
│   ├── test_loader.py           # 대회 측 제공 복호화 로더 (수정 금지)
│   ├── imaging.py               # 분류 입력용 색상 제거 (그레이스케일 + Sauvola/Otsu 이진화)
│   ├── preprocessing.py         # ① train patch 추출 (색상은 ink mask 탐색에만 사용)
│   ├── embedding.py             # ② frozen DINOv2/CLIP 임베딩 추출
│   ├── train.py                 # ③ leakage-safe CV + class-weighted 분류기
│   └── inference.py             # ④ submission.csv 생성
├── colab_pipeline.ipynb         # Colab에서 ①~④ 실행
└── patches_cache/, embeddings/, models/   # 각 단계 산출물 (재실행 시 캐시 재사용)
```

## 왜 이런 구조인가

- **test 는 이미 크롭·암호화되어 제공**된다 (id별 독립 채점, 원본 페이지 grouping 정보 없음).
  그래서 손글씨 영역 추출(①)은 **train 원본 페이지에만** 적용하고, test 는
  `src/test_loader.TestDS` 로 복호화한 512×512 이미지를 그대로 임베딩 입력으로 쓴다.
- **train/val 분할은 페이지 단위**(`StratifiedGroupKFold`, `groups=page_id`)로 한다.
  같은 페이지에서 나온 patch 를 train/val에 나눠 넣으면 val 점수가 과도하게 낙관적으로
  나오는 leakage가 생기기 때문이다.
- 주 지표는 **patch-level Macro F1**이다. test는 크롭 하나당 독립 채점이라, 페이지 단위로
  여러 patch 예측을 모으는 방식은 test에는 적용할 수 없다 — patch 단위 성능이 리더보드와
  가장 가까운 proxy 다. (참고용으로 페이지 단위 지표도 함께 출력한다.)
- **색상은 extraction(어디에 손글씨가 있는지 찾기)에만 쓰고, 분류기 입력에서는 제거한다.**
  `preprocessing.py`는 HSV 채도로 색 잉크를, 그레이스케일 명도로 검정 잉크를 찾아 patch
  위치를 정하지만, 저장되는 patch 자체는 `src/imaging.py`를 거쳐 그레이스케일(+Sauvola
  이진화)로 바뀐다. test 쪽도 `embedding.py`의 `raw_test_transform()`이 복호화된 컬러
  이미지를 백본에 넣기 전에 동일하게 변환한다. 어떤 펜을 들었는지는 작성자의 안정적인
  특징이 아니라 그날그날 바뀌는 우연적 변수라서, 색을 남겨두면 소량 데이터에서 모델이
  진짜 필체 대신 색상을 지름길로 학습할 위험이 있기 때문이다.

## 로컬 실행 (개발/디버그, CPU 소량)

```bash
pip install torch torchvision pillow numpy pandas scikit-learn scikit-image joblib open_clip_torch opencv-python-headless

python -m src.preprocessing --limit-pages 3
python -m src.embedding --split train --backbone dinov2 --device cpu --limit 20
python -m src.embedding --split test  --backbone dinov2 --device cpu --limit 5
python -m src.train --backbone dinov2 --n-splits 2
python -m src.inference --backbone dinov2
```

`--limit-pages`/`--limit`는 로컬에서 배선(wiring)만 빠르게 확인하기 위한 옵션이다.
실제 성능 실험/제출용 학습은 Colab에서 전체 데이터로 돌린다.

## Colab 실행 (실제 학습/제출)

1. `colab_pipeline.ipynb`를 Colab에서 열고 런타임을 GPU(T4 이상)로 설정.
2. `12-aikuthon/` 전체(이 저장소, train/test 데이터 포함)를
   `/content/drive/MyDrive/writer_id_project/`에 업로드해 둔다.
3. 노트북 셀을 순서대로 실행 (0. Drive 마운트 → 1. 의존성 설치 → ①~④).
4. 임베딩(`embeddings/`)과 분류기(`models/`)는 Drive에 저장되므로, 세션이 끊겨도
   ①·②를 다시 돌릴 필요 없이 이어서 실행할 수 있다.

## 대회 규칙 — 반드시 지킬 것

- **test 복호화 결과를 화면에 출력하거나 파일로 저장하거나 사람이 보고 라벨을 붙이면 실격**이다.
  `src/embedding.py`, `src/inference.py`의 test 경로는 텐서만 다루고, 어디에도 이미지
  저장/출력 코드가 없다 — 코드를 수정할 때도 이 경로에는 `imshow`/`.save(`/`print(image)` 류를
  추가하지 않는다.
- `src/test_loader.py`의 `_PART_A/B/C` 키 재료는 절대 수정하지 않는다.
- 사전학습 백본은 **frozen**으로만 사용한다 (전체 파인튜닝 금지). 외부 데이터셋 사용 금지.
  test 데이터는 학습/정규화 통계, 하이퍼파라미터 튜닝 어디에도 사용하지 않는다 (leakage 금지).
- 제출은 최대 30회 — 로컬/Colab CV 점수로 먼저 검증한 뒤에만 제출한다.

## 다음 단계 (이번 MVP 범위 밖)

- `features.py` — contour 기반 handcrafted 특징, OCR 슬라이드 제목 메타정보.
  메타정보는 train의 "과목-작성자" 상관관계가 test에서 깨질 수 있으므로 반드시
  ablation(켰을 때/껐을 때 CV 비교)으로 검증한 뒤 채택할 것.
- `ensemble.py` — DINOv2 + CLIP(+handcrafted) 예측 확률의 soft-voting/stacking 결합.
- backbone 비교(DINOv2 vs CLIP), class-imbalance 보정 강화(추가 augmentation,
  `WeightedRandomSampler`), 마지막 1~2개 transformer block만 unfreeze하는 부분 파인튜닝 실험.
