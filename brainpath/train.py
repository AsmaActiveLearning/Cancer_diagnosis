"""Train and evaluate the pilot on saved Virchow features. Runs on a laptop CPU.

Three results, one per research question in the pilot:
  RQ1 common types - patient-grouped cross-validation of a slide classifier
  RQ1 rare types   - leave-one-out nearest-prototype recognition
  RQ2 unseen types - AUROC for telling held-out types apart from known ones

Two slide classifiers are compared:
  mean  - mean of tile features + logistic regression (baseline)
  abmil - gated attention multiple-instance learning (Ilse et al., 2018);
          its attention weights are saved for the explanation step

Example:
  python -m brainpath.train features/ --out results/ --patient-col patient_id
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn


def load_features(feature_dir, patient_col=None):
    slides = []
    for f in sorted(Path(feature_dir).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        meta = d.get("meta", {})
        slides.append({
            "slide_id": d["slide_id"],
            "tumour_type": d["tumour_type"],
            "role": d["role"],
            "patient": str(meta.get(patient_col, d["slide_id"])) if patient_col else d["slide_id"],
            "features": d["features"].float(),
        })
    if not slides:
        raise SystemExit(f"no .pt feature files in {feature_dir}")
    return slides


def l2norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


class ABMIL(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(hidden, 128), nn.Sigmoid())
        self.att_w = nn.Linear(128, 1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.proj(x)
        a = torch.softmax(self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1), dim=0)
        return self.head(a @ h), a


def train_abmil(bags, labels, n_classes, epochs=30, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    model = ABMIL(bags[0].shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for i in rng.permutation(len(bags)):
            logits, _ = model(bags[i])
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([labels[i]]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def abmil_predict(model, bags):
    probs, atts = [], []
    for b in bags:
        logits, a = model(b)
        probs.append(torch.softmax(logits, -1).numpy())
        atts.append(a.numpy())
    return np.stack(probs), atts


def run(slides, method="abmil", n_splits=5, epochs=30, seed=0):
    common = [s for s in slides if s["role"] == "common"]
    rare = [s for s in slides if s["role"] == "rare"]
    heldout = [s for s in slides if s["role"] == "heldout"]
    classes = sorted({s["tumour_type"] for s in common})
    y = np.array([classes.index(s["tumour_type"]) for s in common])
    groups = [s["patient"] for s in common]
    pooled = {s["slide_id"]: l2norm(s["features"].mean(0).numpy()) for s in slides}

    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise SystemExit("each common type needs at least 2 slides for cross-validation")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows, attention = [], {}
    heldout_msp = np.zeros((len(heldout), n_splits))
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        train_s, test_s = [common[i] for i in tr], [common[i] for i in te]
        if method == "mean":
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(np.stack([pooled[s["slide_id"]] for s in train_s]), y[tr])
            def predict(ss):
                return clf.predict_proba(np.stack([pooled[s["slide_id"]] for s in ss])), None
        else:
            model = train_abmil([s["features"] for s in train_s], y[tr], len(classes), epochs, seed=seed + fold)
            def predict(ss):
                return abmil_predict(model, [s["features"] for s in ss])

        probs, atts = predict(test_s)
        for k, s in enumerate(test_s):
            rows.append({"slide_id": s["slide_id"], "role": "common", "true": s["tumour_type"],
                         "pred": classes[probs[k].argmax()], "msp": float(probs[k].max()), "fold": fold})
            if atts is not None:
                attention[s["slide_id"]] = atts[k]
        if heldout:
            heldout_msp[:, fold] = predict(heldout)[0].max(1)

    # Prototypes: one centroid per known type (common + rare) from mean-pooled slide embeddings.
    known = common + rare
    def centroids(exclude=None):
        types = sorted({s["tumour_type"] for s in known})
        cents = {}
        for t in types:
            members = [pooled[s["slide_id"]] for s in known if s["tumour_type"] == t and s["slide_id"] != exclude]
            if members:
                cents[t] = l2norm(np.mean(members, 0))
        return cents

    def nearest(slide_id, exclude=None):
        cents = centroids(exclude)
        names = list(cents)
        sims = np.stack([cents[t] for t in names]) @ pooled[slide_id]
        return names[int(sims.argmax())], float(sims.max())

    for s in rare:  # leave this slide out of its own prototype
        pred, sim = nearest(s["slide_id"], exclude=s["slide_id"])
        rows.append({"slide_id": s["slide_id"], "role": "rare", "true": s["tumour_type"], "pred": pred,
                     "proto_sim": sim})
    for r in rows:
        if r["role"] == "common":
            r["proto_sim"] = nearest(r["slide_id"], exclude=r["slide_id"])[1]
    for i, s in enumerate(heldout):
        pred, sim = nearest(s["slide_id"])
        rows.append({"slide_id": s["slide_id"], "role": "heldout", "true": s["tumour_type"],
                     "pred": pred, "msp": float(heldout_msp[i].mean()), "proto_sim": sim})

    preds = pd.DataFrame(rows)
    c = preds[preds.role == "common"]
    r = preds[preds.role == "rare"]
    summary = {
        "method": method,
        "n_slides": {"common": len(common), "rare": len(rare), "heldout": len(heldout)},
        "common_balanced_accuracy": float(balanced_accuracy_score(c.true, c.pred)),
        "rare_accuracy": float((r.true == r.pred).mean()) if len(r) else None,
    }
    if heldout:
        known_vs_unseen = preds[preds.role.isin(["common", "heldout"])]
        is_unseen = (known_vs_unseen.role == "heldout").astype(int)
        # Low confidence / low similarity should mean "unseen", so negate the scores.
        summary["unseen_auroc_msp"] = float(roc_auc_score(is_unseen, -known_vs_unseen.msp))
        summary["unseen_auroc_prototype"] = float(roc_auc_score(is_unseen, -known_vs_unseen.proto_sim))
    return summary, preds, attention


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results")
    p.add_argument("--method", choices=["mean", "abmil", "both"], default="both")
    p.add_argument("--patient-col", default=None, help="manifest column with the patient id, if any")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    slides = load_features(args.features, args.patient_col)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = ["mean", "abmil"] if args.method == "both" else [args.method]
    summaries = []
    for m in methods:
        summary, preds, attention = run(slides, m, args.folds, args.epochs, args.seed)
        preds.to_csv(out / f"predictions_{m}.csv", index=False)
        if attention:
            np.savez_compressed(out / f"attention_{m}.npz", **{Path(k).stem: v for k, v in attention.items()})
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
