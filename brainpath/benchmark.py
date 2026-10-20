"""Compare slide classifiers on the known types. Runs on a laptop CPU in a minute or two.

Every slide of a known type (role common or rare) is predicted by a model fit
without it and without any other slide of the same patient (leave-one-patient-out),
so the numbers are honest even with 4-5 slides per type.

Slide embeddings compared:
  mean      - mean of the tile embeddings (what report.py and explain.py use)
  mean+max  - mean and max of the tile embeddings, concatenated
Classifiers compared:
  logreg_l2_C1   - logistic regression on L2-normalised embeddings, C=1 (current default)
  logreg_std     - standardised features, C picked by inner cross-validation
  pca_logreg     - standardised, PCA to 32 dimensions, then logistic regression
  knn1           - nearest known slide by cosine similarity
  centroid       - nearest type centroid by cosine similarity
  svm_linear     - linear SVM on standardised features

Reported: accuracy, balanced accuracy (each type counts the same), and the mean
top probability where the model gives one (a well-behaved model is confident
when right).

Example:
  python -m brainpath.benchmark features/ --out results/benchmark/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from brainpath.explore import load
from brainpath.train import l2norm

L2 = FunctionTransformer(l2norm)


def embeddings(slides, kind):
    if kind == "mean":
        return np.stack([s["features"].mean(0) for s in slides]).astype(np.float64)
    if kind == "mean+max":
        return np.stack([np.r_[s["features"].mean(0), s["features"].max(0)] for s in slides]).astype(np.float64)
    raise ValueError(kind)


def models(min_class):
    inner = max(2, min(3, min_class - 1))
    return {
        "logreg_l2_C1": make_pipeline(L2, LogisticRegression(C=1.0, max_iter=3000)),
        "logreg_std": make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=inner, max_iter=3000)),
        "pca_logreg": make_pipeline(StandardScaler(), PCA(n_components=32, random_state=0),
                                    LogisticRegression(C=0.1, max_iter=3000)),
        "knn1": make_pipeline(L2, KNeighborsClassifier(n_neighbors=1)),
        "centroid": make_pipeline(L2, NearestCentroid()),
        "svm_linear": make_pipeline(StandardScaler(), LinearSVC(C=1e-3, max_iter=20000)),
    }


def evaluate(slides, seed=0):
    known = [s for s in slides if s["role"] != "heldout"]
    y = np.array([s["tumour_type"] for s in known])
    groups = np.array([s["meta"].get("pat_id", s["slide_id"]) for s in known])
    min_class = int(pd.Series(y).value_counts().min())
    rows, preds = [], []
    for kind in ("mean", "mean+max"):
        X = embeddings(known, kind)
        for name, model in models(min_class).items():
            pred = np.empty(len(y), dtype=object)
            top = np.full(len(y), np.nan)
            for tr, te in LeaveOneGroupOut().split(X, y, groups):
                pca_dim = min(32, len(tr) - 1)
                if name == "pca_logreg":
                    model.set_params(pca__n_components=pca_dim)
                if name == "logreg_std":
                    counts = pd.Series(y[tr]).value_counts().min()
                    model.set_params(logisticregressioncv__cv=max(2, min(3, int(counts))))
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])
                if hasattr(model, "predict_proba"):
                    top[te] = model.predict_proba(X[te]).max(1)
            right = pred == y
            rows.append({"embedding": kind, "classifier": name,
                         "accuracy": right.mean(), "balanced_accuracy": balanced_accuracy_score(y, pred),
                         "mean_top_prob_when_right": np.nanmean(top[right]) if right.any() else np.nan,
                         "mean_top_prob_when_wrong": np.nanmean(top[~right]) if (~right).any() else np.nan})
            preds += [{"embedding": kind, "classifier": name, "slide_id": s["slide_id"],
                       "true": t, "pred": p} for s, t, p in zip(known, y, pred)]
    table = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False, ignore_index=True)
    return table, pd.DataFrame(preds)


def per_type(preds, embedding, classifier):
    d = preds[(preds.embedding == embedding) & (preds.classifier == classifier)]
    return d.assign(right=d.true == d.pred).groupby("true").right.agg(["sum", "count"]) \
        .rename(columns={"sum": "right", "count": "slides"})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/benchmark")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    table, preds = evaluate(load(args.features))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "benchmark.csv", index=False)
    preds.to_csv(out / "benchmark_predictions.csv", index=False)
    best = table.iloc[0]
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(table.to_string(index=False))
        print(f"\nbest: {best.classifier} on {best.embedding}, per type:")
        print(per_type(preds, best.embedding, best.classifier).to_string())


if __name__ == "__main__":
    main()
