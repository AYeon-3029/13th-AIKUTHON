# 13th AIKUTHON — Writer Identification

제13회 AIKUTHON 필적 작성자 분류 프로젝트입니다. 강의 슬라이드에 적힌 손글씨를 바탕으로 작성자를 구분하며, 수동 크롭·획 두께 정규화와 여러 이미지 백본을 실험했습니다.

## Final Submission

| 자료 | 파일 |
| --- | --- |
| 최종 제출 코드 | [final_submission.ipynb](final_submission/final_submission.ipynb) |
| 최종 발표 자료 | [final_presentation.pdf](final_submission/final_presentation.pdf) |

최종 제출 노트북과 발표 자료는 제출 당시 원본을 보존했습니다. 노트북에 기록된 중간 실험 점수는 최종 수상 결과와 구분해서 확인해주세요.

## Awards

수상 증빙: [13th AIKUTHON 상장 PDF](awards/13th_aikuthon_certificate.pdf)

## Repository Structure

```text
13th-AIKUTHON/
├── final_submission/       # 최종 제출 노트북 및 발표 PDF
├── awards/                 # 수상 증빙
├── src/                    # 전처리·임베딩·학습·추론 모듈
├── teammate/               # 팀 실험 노트북과 crop 매핑
├── docs/                   # 대회 개요·데이터 설명·규칙·파이프라인 문서
├── *.ipynb                 # baseline 및 Colab 실험 노트북
└── sample_submission.csv   # 제출 형식 예시
```

## Getting Started

기본 모듈 파이프라인은 [colab_pipeline.ipynb](colab_pipeline.ipynb), 최종 제출 모델은 [최종 제출 노트북](final_submission/final_submission.ipynb)을 참고하세요. 각 노트북의 데이터 경로를 실행 환경에 맞게 설정해야 합니다.

```bash
pip install torch torchvision pillow numpy pandas scikit-learn scikit-image joblib open_clip_torch opencv-python-headless
```

원본 학습·평가 데이터와 수동 크롭 데이터는 저장소에 포함하지 않습니다. 대회 데이터를 별도로 준비해 기존 코드가 사용하는 `train/train/`, `test/`, `crop/` 경로에 배치하세요. 최종 노트북에서 사용하는 `crop_labeling.zip` 역시 별도로 준비해야 합니다.

## Documentation

- [기본 파이프라인 및 실행 방법](docs/pipeline.md)
- [대회 개요](docs/overview.md)
- [데이터 설명](docs/data_description.md)
- [대회 규칙](docs/rules.md)

실험 코드와 제출 당시 기록을 함께 보관한 아카이브입니다. 기본 파이프라인 문서는 초기 접근 방식의 설명이며, 최종 제출 방식은 최종 노트북과 발표 자료를 기준으로 확인하세요.
