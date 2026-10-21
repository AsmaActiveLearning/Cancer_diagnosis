"""Detect unseen tumour types from saved Virchow features. Runs on a laptop CPU.

Each slide becomes one embedding (mean of its tile features). A detector is fit
on the known types only, and gives every slide an "unseen score": the higher
the score, the less the slide looks like any known type. Five scores are compared:
  msp         - 1 minus the classifier's top probability (Hendrycks & Gimpel, 2017)
  energy      - negative log-sum-exp of the classifier logits (Liu et al., 2020)
  mahalanobis - distance to the nearest class mean, shared covariance (Lee et al., 2018)
  knn         - cosine distance to the k-th nearest known slide (Sun et al., 2022)
  prototype   - 1 minus cosine similarity to the nearest class centroid

Evaluation hides tumour types in rotation. By default every rare and held-out
type is hidden once, while all others are known. Known slides are scored
out-of-fold (patient-grouped cross-validation), so no slide is scored by a
model that saw it.

Reported per score, as mean and standard deviation over rotations:
  auroc            - how well the score separates unseen from known slides
  fpr_at_95tpr     - share of known slides flagged when 95% of unseen ones are caught
  unseen_detected  - share of unseen slides flagged at the known-only threshold
  known_flagged    - share of known slides flagged at that threshold
The threshold is set from known slides only (so that --known-keep of them are
accepted), which is how it would be set for slides of truly new types.

Example:
  python -m brainpath.openset features/ --out results/openset/ --patient-col pat_id

By default every score is computed twice: on L2-normalised embeddings, and on
embeddings standardised per dimension first (--prep l2/std/both).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from brainpath.train import l2norm, load_features

SCORES = ("msp", "energy", "mahalanobis", "knn", "prototype")


class OpenSetDetector:
    """Fit on known slide embeddings; score() returns unseen scores (higher = more likely unseen)."""

    def __init__(self, knn_k=1, pca_dim=32, C=1.0, seed=0, standardize=False):
        self.knn_k, self.pca_dim, self.C, self.seed, self.standardize = knn_k, pca_dim, C, seed, standardize

    def prep(self, X):
        """Optionally standardise each dimension (fit on known slides), then L2-normalise."""
        X = np.asarray(X, dtype=np.float64)
        if self.standardize:
            X = self.scaler_.transform(X)
        return l2norm(X)

    def fit(self, X, y):
        if self.standardize:
            self.scaler_ = StandardScaler().fit(np.asarray(X, dtype=np.float64))
        X = self.prep(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 known types")
        self.clf_ = LogisticRegression(max_iter=2000, C=self.C).fit(X, y)
        self.train_ = X
        self.centroids_ = l2norm(np.stack([X[y == c].mean(0) for c in self.classes_]))
        # Mahalanobis in a small PCA space: 2560 dimensions is far too many for a few dozen slides.
        dim = max(1, min(self.pca_dim, len(X) - len(self.classes_), X.shape[1]))
        self.pca_ = PCA(n_components=dim, random_state=self.seed).fit(X)
        Z = self.pca_.transform(X)
        self.means_ = np.stack([Z[y == c].mean(0) for c in self.classes_])
        resid = Z - self.means_[np.searchsorted(self.classes_, y)]
        self.precision_ = LedoitWolf().fit(resid).precision_
        return self

    def predict(self, X):
        return self.clf_.predict(self.prep(X))

    def score(self, X):
        X = self.prep(X)
        logits = self.clf_.decision_function(X)
        if logits.ndim == 1:  # two known types: sklearn returns one logit
            logits = np.stack([np.zeros_like(logits), logits], 1)
        probs = self.clf_.predict_proba(X)
        diff = self.pca_.transform(X)[:, None, :] - self.means_[None]
        maha = np.einsum("nkd,de,nke->nk", diff, self.precision_, diff).min(1)
        dist = 1 - X @ self.train_.T
        k = min(self.knn_k, dist.shape[1])
        return {
            "msp": 1 - probs.max(1),
            "energy": -logsumexp(logits, axis=1),
            "mahalanobis": maha,
            "knn": np.sort(dist, axis=1)[:, k - 1],
            "prototype": 1 - (X @ self.centroids_.T).max(1),
        }


def fpr_at_tpr(is_unseen, score, tpr=0.95):
    fpr, tprs, _ = roc_curve(is_unseen, score)
    return float(fpr[np.searchsorted(tprs, tpr)])


def evaluate_split(slides, unseen_types, n_splits=5, known_keep=0.95, seed=0, **detector_kw):
    """Hide unseen_types, score every slide, and return (per-slide rows, metrics per score)."""
    known = [s for s in slides if s["tumour_type"] not in unseen_types]
    unseen = [s for s in slides if s["tumour_type"] in unseen_types]
    if not unseen:
        raise ValueError(f"no slides of {unseen_types}")
    emb = {s["slide_id"]: s["features"].mean(0).numpy() for s in slides}
    Xk = np.stack([emb[s["slide_id"]] for s in known])
    yk = np.array([s["tumour_type"] for s in known])
    Xu = np.stack([emb[s["slide_id"]] for s in unseen])
    groups = [s["patient"] for s in known]

    n_splits = min(n_splits, int(pd.Series(yk).value_counts().min()))
    if n_splits < 2:
        raise ValueError("every known type needs at least 2 slides")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    known_scores = {name: np.zeros(len(known)) for name in SCORES}
    known_pred = np.empty(len(known), dtype=object)
    unseen_scores = {name: np.zeros(len(unseen)) for name in SCORES}
    for tr, te in cv.split(Xk, yk, groups):
        det = OpenSetDetector(seed=seed, **detector_kw).fit(Xk[tr], yk[tr])
        for name, v in det.score(Xk[te]).items():
            known_scores[name][te] = v
        known_pred[te] = det.predict(Xk[te])
        for name, v in det.score(Xu).items():
            unseen_scores[name] += v / n_splits

    rows = [{"slide_id": s["slide_id"], "tumour_type": s["tumour_type"], "status": "known",
             "pred": known_pred[i], **{n: known_scores[n][i] for n in SCORES}} for i, s in enumerate(known)]
    rows += [{"slide_id": s["slide_id"], "tumour_type": s["tumour_type"], "status": "unseen",
              "pred": None, **{n: unseen_scores[n][i] for n in SCORES}} for i, s in enumerate(unseen)]

    is_unseen = np.r_[np.zeros(len(known)), np.ones(len(unseen))]
    metrics = {}
    for name in SCORES:
        score = np.r_[known_scores[name], unseen_scores[name]]
        threshold = float(np.quantile(known_scores[name], known_keep, method="higher"))
        metrics[name] = {
            "auroc": float(roc_auc_score(is_unseen, score)),
            "fpr_at_95tpr": fpr_at_tpr(is_unseen, score),
            "unseen_detected": float((unseen_scores[name] > threshold).mean()),
            "known_flagged": float((known_scores[name] > threshold).mean()),
            "threshold": threshold,
        }
    metrics["known_accuracy"] = float((known_pred == yk).mean())
    return rows, metrics


def rotations(slides, mode="rare+heldout"):
    """Which types to hide, one list per rotation."""
    by_role = {}
    for s in slides:
        by_role.setdefault(s["role"], set()).add(s["tumour_type"])
    if mode == "heldout":
        return [sorted(by_role.get("heldout", ()))]
    roles = ("rare", "heldout") if mode == "rare+heldout" else ("common", "rare", "heldout")
    return [[t] for r in roles for t in sorted(by_role.get(r, ()))]


def run(slides, mode="rare+heldout", **kw):
    all_rows, per_rotation = [], []
    for hidden in rotations(slides, mode):
        rows, metrics = evaluate_split(slides, hidden, **kw)
        tag = "+".join(hidden)
        all_rows += [{**r, "hidden": tag} for r in rows]
        per_rotation.append({"hidden": tag, **metrics})

    table = []
    for name in SCORES:
        vals = pd.DataFrame([m[name] for m in per_rotation])
        row = {"score": name, "rotations": len(vals)}
        for col in ("auroc", "fpr_at_95tpr", "unseen_detected", "known_flagged"):
            row[f"{col}_mean"] = float(vals[col].mean())
            row[f"{col}_sd"] = float(vals[col].std(ddof=0))
        table.append(row)
    table = pd.DataFrame(table).sort_values("auroc_mean", ascending=False, ignore_index=True)
    return table, pd.DataFrame(all_rows), per_rotation


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/openset")
    p.add_argument("--patient-col", default=None, help="manifest column with the patient id, if any")
    p.add_argument("--rotate", choices=["rare+heldout", "all", "heldout"], default="rare+heldout",
                   help="which types to hide one at a time ('heldout' hides only the held-out types, once)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--known-keep", type=float, default=0.95,
                   help="share of known slides the threshold accepts")
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prep", choices=["l2", "std", "both"], default="both",
                   help="embedding preparation: L2 only, standardise then L2, or compare both")
    args = p.parse_args()

    slides = load_features(args.features, args.patient_col)
    tables, all_scores, per_rotation = [], [], []
    for prep in (["l2", "std"] if args.prep == "both" else [args.prep]):
        t, sc, rot = run(slides, args.rotate, n_splits=args.folds, known_keep=args.known_keep, seed=args.seed,
                         knn_k=args.knn_k, pca_dim=args.pca_dim, standardize=prep == "std")
        tables.append(t.assign(prep=prep))
        all_scores.append(sc.assign(prep=prep))
        per_rotation += [{**r, "prep": prep} for r in rot]
    table = pd.concat(tables).sort_values("auroc_mean", ascending=False, ignore_index=True)
    table = table[["prep"] + [c for c in table.columns if c != "prep"]]
    scores = pd.concat(all_scores, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "openset_summary.csv", index=False)
    scores.to_csv(out / "openset_scores.csv", index=False)
    (out / "openset_rotations.json").write_text(json.dumps(per_rotation, indent=2))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
