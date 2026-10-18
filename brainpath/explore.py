"""Explore the Virchow embeddings before training anything. Runs on a laptop CPU.

Questions it answers, each with a figure or table:
  1. inventory    - how many tiles per slide, and are any slides thin?
  2. pca          - how many directions carry the variance (explained variance)?
  3. map          - do slides of one type sit together? (PCA and UMAP, or t-SNE)
  4. similarity   - cosine similarity between every pair of slides, ordered by type
  5. separability - silhouette per type, leave-one-out 1-NN accuracy on known types,
                    and how far each held-out slide sits from its nearest known type
  6. confounders  - do the main directions track tile count, tissue area or age
                    rather than biology?
  7. tile clusters- k-means on tiles from all slides: which tissue patterns each type has
  8. tile maps    - one slide's tiles coloured by their first 3 principal components

Slide embedding = mean of its tile embeddings, L2-normalised (as in train.py).

Example:
  python -m brainpath.explore features/ --out results/explore/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from brainpath.train import l2norm

# Reference palette (light mode). Roles use the first three categorical slots,
# the most a scatter can carry while staying colour-blind safe.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROLE_COLORS = {"common": "#2a78d6", "rare": "#eb6834", "heldout": "#1baf7a"}
ROLE_ORDER = ["common", "rare", "heldout"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def load(feature_dir):
    """Slides with tile features, coordinates and metadata from extract.py output."""
    slides = []
    for f in sorted(Path(feature_dir).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        slides.append({"slide_id": d["slide_id"], "tumour_type": d["tumour_type"], "role": d["role"],
                       "features": d["features"].float().numpy(),
                       "coords": d["coords"].numpy() if "coords" in d else None,
                       "meta": d.get("meta", {})})
    if not slides:
        raise SystemExit(f"no .pt feature files in {feature_dir}")
    order = {r: i for i, r in enumerate(ROLE_ORDER)}
    return sorted(slides, key=lambda s: (order.get(s["role"], 9), s["tumour_type"], s["slide_id"]))


def slide_table(slides):
    rows = []
    for s in slides:
        m = s["meta"]
        rows.append({"slide_id": s["slide_id"], "tumour_type": s["tumour_type"], "role": s["role"],
                     "n_tiles": len(s["features"]),
                     "patient": m.get("pat_id"), "age": m.get("age"), "sex": m.get("sex"),
                     "tissue_area": m.get("tissue_area"), "cellularity": m.get("cellularity")})
    return pd.DataFrame(rows)


def slide_matrix(slides):
    return l2norm(np.stack([s["features"].mean(0) for s in slides]).astype(np.float64))


def embed_2d(X, method="umap", seed=0):
    """2-D map of slides. UMAP if installed, else t-SNE; 'pca' for the linear view."""
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"
    if method == "umap":
        try:
            import umap
            n = min(15, len(X) - 1)
            return umap.UMAP(n_neighbors=n, metric="cosine", random_state=seed).fit_transform(X), "UMAP"
        except ImportError:
            pass
    from sklearn.manifold import TSNE
    perplexity = max(2, min(10, (len(X) - 1) // 3))
    return TSNE(n_components=2, metric="cosine", perplexity=perplexity, init="pca",
                random_state=seed).fit_transform(X), "t-SNE"


def separability(X, table):
    """Per-slide silhouette (known types), leave-one-out 1-NN, and held-out distances."""
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    known = (table.role != "heldout").to_numpy()
    types = table.tumour_type.to_numpy()

    out = table[["slide_id", "tumour_type", "role"]].copy()
    sil = np.full(len(X), np.nan)
    if known.sum() > 2 and len(set(types[known])) > 1:
        sil[known] = silhouette_samples(X[known], types[known], metric="cosine")
    out["silhouette"] = sil
    # Nearest known slide for every slide (excluding itself).
    sim_known = np.where(known[None, :], sim, -np.inf)
    nn = sim_known.argmax(1)
    out["nearest_known_type"] = types[nn]
    out["nearest_known_sim"] = sim_known.max(1)
    out["nn_correct"] = np.where(known, types[nn] == types, np.nan)

    per_type = out.groupby(["role", "tumour_type"], sort=False).agg(
        slides=("slide_id", "size"),
        silhouette=("silhouette", "mean"),
        nn_accuracy=("nn_correct", "mean"),
        nearest_known_sim=("nearest_known_sim", "mean"),
    ).reset_index()
    summary = {
        "known_1nn_accuracy": float(np.nanmean(out.nn_correct[known])) if known.any() else None,
        "mean_silhouette_known": float(np.nanmean(sil)) if known.sum() > 2 else None,
        "nearest_known_sim_known": float(out.nearest_known_sim[known].mean()) if known.any() else None,
        "nearest_known_sim_heldout": float(out.nearest_known_sim[~known].mean()) if (~known).any() else None,
    }
    return out, per_type, summary


def confounders(X, table, n_components=5):
    """Spearman correlation of the top principal components with non-biological slide traits."""
    pcs = PCA(n_components=min(n_components, len(X) - 1)).fit_transform(X)
    rows = []
    for col in ("n_tiles", "tissue_area", "cellularity", "age"):
        v = pd.to_numeric(table[col], errors="coerce")
        if v.notna().sum() < 5:
            continue
        for k in range(pcs.shape[1]):
            rows.append({"trait": col, "pc": f"PC{k + 1}",
                         "spearman": float(pd.Series(pcs[:, k]).corr(v, method="spearman"))})
    return pd.DataFrame(rows)


def tile_clusters(slides, k=8, per_slide=300, seed=0):
    """k-means on a sample of tiles from every slide; returns cluster share per type."""
    rng = np.random.default_rng(seed)
    feats, owner = [], []
    for i, s in enumerate(slides):
        f = s["features"]
        idx = rng.choice(len(f), size=min(per_slide, len(f)), replace=False)
        feats.append(f[idx])
        owner += [i] * len(idx)
    F = l2norm(np.concatenate(feats).astype(np.float64))
    labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(F)
    df = pd.DataFrame({"type": [slides[i]["tumour_type"] for i in owner], "cluster": labels})
    share = pd.crosstab(df.type, df.cluster, normalize="index")
    order = {s["tumour_type"]: (ROLE_ORDER.index(s["role"]), s["tumour_type"]) for s in slides}
    return share.loc[sorted(share.index, key=order.get)]


def tile_rgb(slide, seed=0):
    """First 3 principal components of one slide's tiles, scaled to 0-1 as RGB."""
    pcs = PCA(n_components=3, random_state=seed).fit_transform(slide["features"])
    lo, hi = np.percentile(pcs, 2, axis=0), np.percentile(pcs, 98, axis=0)
    return np.clip((pcs - lo) / (hi - lo + 1e-8), 0, 1)


# ----------------------------------------------------------------- figures

def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(w=7, h=5):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_map(Y, table, method, path=None):
    """Slides in 2-D: colour = role, marker = type, type names at each type's centre."""
    fig, ax = _figure(8, 6)
    _style(ax, f"Slides by Virchow embedding ({method})", f"{method} 1", f"{method} 2")
    ax.set_xticks([]), ax.set_yticks([])
    for i, (t, grp) in enumerate(table.groupby("tumour_type", sort=False)):
        role = grp.role.iloc[0]
        idx = grp.index.to_numpy()
        ax.scatter(Y[idx, 0], Y[idx, 1], s=46, marker=MARKERS[i % len(MARKERS)],
                   color=ROLE_COLORS.get(role, INK_2), edgecolor=SURFACE, linewidth=1.5, zorder=3)
        cx, cy = Y[idx].mean(0)
        ax.annotate(t if len(t) <= 28 else t[:26] + "…", (cx, cy), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7, color=INK, zorder=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=r) for r, c in ROLE_COLORS.items()
               if r in set(table.role)]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    return _save(fig, path)


def plot_explained_variance(X, path=None):
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = _figure(6, 3.6)
    _style(ax, "Variance explained by principal components", "number of components", "cumulative share")
    ax.plot(np.arange(1, len(cum) + 1), cum, color=ROLE_COLORS["common"], linewidth=2)
    for target in (0.5, 0.8, 0.9):
        n = int(np.searchsorted(cum, target) + 1)
        ax.scatter([n], [cum[n - 1]], s=40, color=ROLE_COLORS["common"], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{int(target * 100)}% at {n}", (n, cum[n - 1]), xytext=(6, -12), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.02)
    return _save(fig, path), cum


def plot_similarity(X, table, path=None):
    fig, ax = _figure(7.5, 6.5)
    sim = X @ X.T
    im = ax.imshow(sim, cmap="Blues", vmin=np.percentile(sim, 5), vmax=1)
    _style(ax, "Cosine similarity between slides (ordered by role, then type)")
    ax.grid(False)
    bounds = np.flatnonzero(table.tumour_type.to_numpy()[1:] != table.tumour_type.to_numpy()[:-1]) + 0.5
    for b in bounds:
        ax.axhline(b, color=SURFACE, linewidth=1.5), ax.axvline(b, color=SURFACE, linewidth=1.5)
    centres = table.groupby("tumour_type", sort=False).apply(lambda g: g.index.to_numpy().mean(), include_groups=False)
    ax.set_yticks(centres.to_numpy(), [t[:30] for t in centres.index], fontsize=7)
    ax.set_xticks([])
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    return _save(fig, path)


def plot_heldout_distance(sep, path=None):
    """Similarity to the nearest known slide: known slides vs held-out slides."""
    fig, ax = _figure(6, 3.6)
    _style(ax, "How close is each slide to its nearest known slide?", "cosine similarity to nearest known slide")
    ax.set_yticks([])
    rng = np.random.default_rng(0)
    for row, role in enumerate(ROLE_ORDER):
        v = sep.nearest_known_sim[sep.role == role].to_numpy()
        if len(v):
            ax.scatter(v, row + rng.uniform(-0.15, 0.15, len(v)), s=40, color=ROLE_COLORS[role],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(role, (v.min(), row), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, 2.6)
    ax.margins(x=0.15)
    return _save(fig, path)


def plot_cluster_share(share, path=None):
    fig, ax = _figure(7.5, 0.4 * len(share) + 1.5)
    im = ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.5, share.to_numpy().max()), aspect="auto")
    _style(ax, "Share of each type's tiles in each tile cluster")
    ax.grid(False)
    ax.set_yticks(range(len(share)), [t[:34] for t in share.index], fontsize=7)
    ax.set_xticks(range(share.shape[1]), [f"C{c}" for c in share.columns], fontsize=8)
    for (i, j), v in np.ndenumerate(share.to_numpy()):
        if v >= 0.15:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color=SURFACE if v > 0.35 else INK)
    return _save(fig, path)


def plot_tile_map(slide, path=None):
    if slide["coords"] is None:
        raise ValueError("this feature file has no tile coordinates")
    rgb, xy = tile_rgb(slide), slide["coords"]
    fig, ax = _figure(5.5, 5.5)
    _style(ax, f"{slide['tumour_type']}: tiles coloured by PCA (RGB = PC1-3)")
    ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
    ax.scatter(xy[:, 0], -xy[:, 1], c=rgb, s=6, marker="s", linewidths=0)
    ax.set_aspect("equal")
    return _save(fig, path)


def _save(fig, path):
    import matplotlib.pyplot as plt
    if path:
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/explore")
    p.add_argument("--clusters", type=int, default=8, help="k for tile k-means")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slides = load(args.features)
    table = slide_table(slides)
    X = slide_matrix(slides)
    table.to_csv(out / "inventory.csv", index=False)

    _, cum = plot_explained_variance(X, out / "pca_variance.png")
    for method in ("pca", "umap"):
        Y, name = embed_2d(X, method, args.seed)
        plot_map(Y, table, name, out / f"map_{name.lower()}.png")
    plot_similarity(X, table, out / "similarity.png")
    sep, per_type, summary = separability(X, table)
    sep.to_csv(out / "separability_slides.csv", index=False)
    per_type.to_csv(out / "separability_types.csv", index=False)
    plot_heldout_distance(sep, out / "nearest_known.png")
    conf = confounders(X, table)
    conf.to_csv(out / "confounders.csv", index=False)
    share = tile_clusters(slides, args.clusters, seed=args.seed)
    share.to_csv(out / "tile_clusters.csv")
    plot_cluster_share(share, out / "tile_clusters.png")
    for t, grp in table.groupby("tumour_type", sort=False):  # one example map per type
        s = slides[grp.index[grp.n_tiles.argmax()]]
        if s["coords"] is not None:
            plot_tile_map(s, out / f"tilemap_{''.join(c if c.isalnum() else '_' for c in t)[:40]}.png")

    summary.update({
        "slides": len(slides), "types": int(table.tumour_type.nunique()),
        "components_for_80pct_variance": int(np.searchsorted(cum, 0.8) + 1),
        "max_abs_confounder_spearman": float(conf.spearman.abs().max()) if len(conf) else None,
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 160, "display.precision", 2):
        print(per_type.to_string(index=False))


if __name__ == "__main__":
    main()
