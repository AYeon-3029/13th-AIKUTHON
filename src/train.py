"""③ patch-level 임베딩으로 class-weighted 분류기를 학습하고 CV 성능을 리포트한다.

leakage 방지: StratifiedGroupKFold 로 페이지(page_id) 단위 그룹을 유지한 채
클래스 비율도 맞춰서 분할한다 (같은 페이지의 patch 가 train/val 양쪽에 섞이지 않음).

주 지표는 patch-level Macro F1 이다 — test 는 페이지 grouping 없이 크롭 하나씩
독립 채점되므로, patch 단위 성능이 리더보드와 가장 가까운 proxy 다.
페이지 단위(패치 확률 평균 후 argmax) 지표는 참고용으로만 함께 출력한다.
"""

import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config


def load_train_embeddings(backbone: str):
    path = config.EMB_DIR / f"{backbone}_train.npz"
    if not path.is_file():
        raise RuntimeError(f"{path} 가 없습니다. embedding.py --split train 을 먼저 실행하세요.")
    data = np.load(path, allow_pickle=True)
    return data["vectors"], data["writer_label"], data["page_id"]


def _page_level_eval(y_val, page_ids_val, proba_val, n_classes):
    """같은 페이지의 patch 확률을 평균해 페이지 단위 예측을 만든다 (참고 지표)."""
    pages = {}
    for i, pid in enumerate(page_ids_val):
        pages.setdefault(pid, {"proba": np.zeros(n_classes), "n": 0, "label": y_val[i]})
        pages[pid]["proba"] += proba_val[i]
        pages[pid]["n"] += 1
    y_true = np.array([v["label"] for v in pages.values()])
    y_pred = np.array([np.argmax(v["proba"] / v["n"]) for v in pages.values()])
    return y_true, y_pred


def cross_validate(vectors, labels, page_ids, n_splits: int, c_grid: list[float]):
    n_classes = len(config.WRITERS)
    results = {c: {"patch_f1": [], "page_f1": []} for c in c_grid}

    for c in c_grid:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(vectors, labels, groups=page_ids)):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            clf.fit(vectors[tr_idx], labels[tr_idx])

            y_val = labels[va_idx]
            y_pred_patch = clf.predict(vectors[va_idx])

            patch_f1 = f1_score(y_val, y_pred_patch, average="macro", zero_division=0)
            patch_acc = accuracy_score(y_val, y_pred_patch)

            # 학습 fold에 없던 클래스는 predict_proba 열이 아예 없으므로, 전체 클래스
            # 폭(n_classes)으로 0-padding 해서 페이지 단위 확률 평균에 안전하게 쓴다.
            proba_val = np.zeros((len(va_idx), n_classes))
            proba_val[:, clf.classes_] = clf.predict_proba(vectors[va_idx])

            y_true_page, y_pred_page = _page_level_eval(y_val, page_ids[va_idx], proba_val, n_classes)
            page_f1 = f1_score(y_true_page, y_pred_page, average="macro", zero_division=0)

            results[c]["patch_f1"].append(patch_f1)
            results[c]["page_f1"].append(page_f1)

            print(f"C={c:<8} fold={fold}  patch_acc={patch_acc:.4f}  "
                  f"patch_macro_f1={patch_f1:.4f}  page_macro_f1={page_f1:.4f}")

            if fold == n_splits - 1:
                cm = confusion_matrix(y_val, y_pred_patch, labels=list(range(n_classes)))
                print(f"  confusion matrix (patch-level, C={c}, last fold):")
                print(f"  {config.WRITERS}")
                print(cm)

    for c in c_grid:
        mean_patch_f1 = float(np.mean(results[c]["patch_f1"]))
        mean_page_f1 = float(np.mean(results[c]["page_f1"]))
        print(f"C={c:<8} mean patch_macro_f1={mean_patch_f1:.4f}  mean page_macro_f1={mean_page_f1:.4f}")

    best_c = max(c_grid, key=lambda c: np.mean(results[c]["patch_f1"]))
    print(f"\n선택된 C={best_c} (patch-level mean Macro F1 기준)")
    return best_c


def fit_final(vectors, labels, best_c: float, backbone: str):
    clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000)
    clf.fit(vectors, labels)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MODEL_DIR / f"{backbone}_logreg.joblib"
    joblib.dump({"model": clf, "backbone": backbone, "c": best_c, "writers": config.WRITERS}, out_path)
    print(f"최종 분류기 저장: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="class-weighted 분류기 학습 + leakage-safe CV")
    ap.add_argument("--backbone", choices=["dinov2", "clip"], default="dinov2")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = ap.parse_args()

    config.set_seed()
    vectors, labels, page_ids = load_train_embeddings(args.backbone)

    pages_per_class = [len(set(page_ids[labels == c])) for c in range(len(config.WRITERS))]
    n_splits = min(args.n_splits, min(pages_per_class))
    if n_splits < args.n_splits:
        print(f"경고: 가장 적은 클래스의 표본 수가 부족해 n_splits 를 {args.n_splits} -> {n_splits} 로 줄입니다.")

    best_c = cross_validate(vectors, labels, page_ids, n_splits, args.c_grid)
    fit_final(vectors, labels, best_c, args.backbone)


if __name__ == "__main__":
    main()
