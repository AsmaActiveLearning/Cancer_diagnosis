"""Which tiles, and which pixels in them, made the model call a slide's tumour type.

The slide embedding is the mean of its tile embeddings (L2-normalised) and the
classifier is linear, so the score of a type splits exactly into one term per
tile:
    logit_c - mean_k logit_k  =  sum_i  (w_c - mean_k w_k) . t_i / (n |mean t|)  +  bias term
Each term is that tile's contribution: positive tiles push the slide towards
the type, negative ones away from it. For a linear model on a mean this is the
exact Shapley value of each tile; no sampling or approximation is involved.
The classifier is refit without the slide being explained (leave-one-out), as in
report.py.

One figure per slide, four panels:
  A  slide thumbnail (or the tile grid) with each tile coloured by contribution
  B  every tile's embedding in 2-D (UMAP), coloured the same way
  C  the real H&E tiles that contributed most
  D  Virchow's attention inside the top tiles: which pixels the model looked at
Panels A-B need only the feature files (laptop). C needs the slide file
(--slides), and D also needs Virchow and a GPU (Kaggle) unless --no-attention.

Examples:
  laptop:  python -m brainpath.explain features/ --out results/explain/
  Kaggle:  python -m brainpath.explain features/ --out results/explain/ --slides /kaggle/tmp/ --download --only <slide_id>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brainpath.explore import INK, INK_2, SURFACE, _save, _style, embed_2d, load
from brainpath.openset import OpenSetDetector
from brainpath.train import l2norm

# Diverging blue <-> red from the reference palette; grey midpoint means "no effect".
DIVERGING = ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#a32524"]


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("contribution", DIVERGING)


def tile_contributions(slides, i, target=None, seed=0):
    """Exact per-tile contributions to one slide's score for `target` (default: its true type).

    Returns (contributions, info). contributions sums, with info["bias"], to the
    slide's relative logit for the target type.
    """
    known = np.array([s["role"] != "heldout" for s in slides])
    mask = known.copy()
    mask[i] = False
    emb = l2norm(np.stack([s["features"].mean(0) for s in slides]).astype(np.float64))
    types = np.array([s["tumour_type"] for s in slides])
    det = OpenSetDetector(seed=seed).fit(emb[mask], types[mask])
    clf, classes = det.clf_, list(det.classes_)

    probs = clf.predict_proba(emb[i:i + 1])[0]
    pred = classes[int(np.argmax(probs))]
    target = target or slides[i]["tumour_type"]
    if target not in classes:  # a held-out type has no weights: explain the prediction instead
        target = pred
    c = classes.index(target)
    W, b = clf.coef_, clf.intercept_
    if W.shape[0] == 1:  # two classes: sklearn keeps one row
        W, b = np.vstack([-W[0] / 2, W[0] / 2]), np.array([-b[0] / 2, b[0] / 2])
    w = W[c] - W.mean(0)

    T = slides[i]["features"].astype(np.float64)
    scale = len(T) * np.linalg.norm(T.mean(0))
    contrib = T @ w / scale
    bias = float(b[c] - b.mean())
    check = float(emb[i] @ w + bias)
    assert np.isclose(contrib.sum() + bias, check), "contributions must add up to the score"
    return contrib, {"target": target, "pred": pred, "prob_target": float(probs[c]),
                     "prob_pred": float(probs.max()), "bias": bias, "score": check,
                     "positive_share": float((contrib > 0).mean())}


def top_tiles(contrib, k=8):
    return np.argsort(contrib)[::-1][:k]


# ----------------------------------------------------------------- slide access

def find_slide(slide_dir, slide_id):
    if not slide_dir:
        return None
    stem = Path(str(slide_id)).stem
    hits = sorted(Path(slide_dir).glob(f"{stem}*"))
    return str(hits[0]) if hits else None


def read_tiles(path, coords, step, size=224):
    """Real H&E tiles at the coordinates Virchow saw, plus a slide thumbnail and its scale."""
    import openslide
    from PIL import Image

    slide = openslide.OpenSlide(path)
    level = slide.get_best_level_for_downsample(step / size)
    ds = slide.level_downsamples[level]
    read = int(np.ceil(step / ds))
    tiles = [slide.read_region((int(x), int(y)), level, (read, read)).convert("RGB").resize((size, size), Image.BILINEAR)
             for x, y in coords]
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((1600, 1600)).convert("RGB")
    slide.close()
    return tiles, thumb, w / thumb.size[0]


# ----------------------------------------------------------------- attention

def attention_maps(model, transform, tiles, device):
    """Class-token attention over patches in Virchow's last block, one 2-D map per tile.

    Averaged over heads; values are normalised to 0-1 per tile.
    """
    import torch

    blk = model.blocks[-1]
    saved = {}
    hook = blk.norm1.register_forward_hook(lambda m, inp, out: saved.__setitem__("x", out))
    batch = torch.stack([transform(t) for t in tiles]).to(device)
    with torch.inference_mode():
        model(batch)
        x = saved["x"]
        B, N, C = x.shape
        attn = blk.attn
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if hasattr(attn, "q_norm"):
            q, k = attn.q_norm(q), attn.k_norm(k)
        a = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1)  # B, heads, N, N
    hook.remove()
    prefix = getattr(model, "num_prefix_tokens", 1)
    cls = a[:, :, 0, prefix:].mean(1).float().cpu().numpy()
    side = int(round(np.sqrt(cls.shape[1])))
    maps = cls.reshape(B, side, side)
    lo, hi = maps.min((1, 2), keepdims=True), maps.max((1, 2), keepdims=True)
    return (maps - lo) / (hi - lo + 1e-12)


# ----------------------------------------------------------------- figure

def plot_explanation(slide, contrib, info, xy2d, tiles=None, thumb=None, thumb_scale=None, maps=None,
                     step=None, k=8, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    cmap = diverging_cmap()
    # Shown as each tile's share of all the evidence, in percent: easier to read than raw logit units.
    contrib = 100 * contrib / (np.abs(contrib).sum() or 1)
    lim = float(np.percentile(np.abs(contrib), 99)) or 1e-9
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    top = top_tiles(contrib, k)
    xy = slide["coords"]

    fig = plt.figure(figsize=(15, 9.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.35, wspace=0.3)

    # A: where on the slide
    ax = fig.add_subplot(grid[0, :2])
    _style(ax, "A  Tile contributions on the slide")
    ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
    if thumb is not None:
        ax.imshow(thumb)
        px = xy / thumb_scale
        s = (step or 1) / thumb_scale
        for (x, y), v in zip(px, contrib):
            ax.add_patch(Rectangle((x, y), s, s, color=cmap(norm(v)), alpha=0.55, linewidth=0))
        for r, j in enumerate(top, 1):
            ax.annotate(str(r), (px[j, 0] + s / 2, px[j, 1] + s / 2), color=INK, fontsize=8, ha="center",
                        va="center", weight="bold")
    else:  # no slide file: draw each tile as a square of its true size on the slide grid
        if step is None:
            gaps = np.diff(np.unique(xy[:, 0]))
            step = int(gaps[gaps > 0].min()) if (gaps > 0).any() else 1
        for (x, y), v in zip(xy, contrib):
            ax.add_patch(Rectangle((x, y), step, step, color=cmap(norm(v)), linewidth=0))
        for r, j in enumerate(top, 1):
            ax.annotate(str(r), xy[j] + step / 2, color=INK, fontsize=8, ha="center", va="center", weight="bold")
        ax.set_xlim(xy[:, 0].min() - step, xy[:, 0].max() + 2 * step)
        ax.set_ylim(xy[:, 1].max() + 2 * step, xy[:, 1].min() - step)
        ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=ax.inset_axes([1.02, 0.1, 0.025, 0.8]), format="%.2f%%")
    cb.ax.set_title("towards\nthe type", color=INK_2, fontsize=7)
    cb.ax.set_xlabel("away", color=INK_2, fontsize=7)
    cb.ax.tick_params(labelsize=7, colors=INK_2)

    # B: where in embedding space
    ax = fig.add_subplot(grid[0, 2:])
    _style(ax, "B  The same tiles in Virchow embedding space (2-D)")
    ax.set_xticks([]), ax.set_yticks([])
    order = np.argsort(np.abs(contrib))
    ax.scatter(xy2d[order, 0], xy2d[order, 1], c=contrib[order], cmap=cmap, norm=norm, s=8, linewidths=0)
    for r, j in enumerate(top, 1):
        ax.annotate(str(r), xy2d[j], xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK, weight="bold")

    # C and D: the tiles themselves and where Virchow looked
    if tiles is not None:
        # With attention: H&E on top, attention overlay below. Without: the tiles in two rows.
        cols = len(tiles) if maps is not None else (len(tiles) + 1) // 2
        cells = grid[1, :].subgridspec(2, cols, hspace=0.12, wspace=0.06)
        for r, (j, img) in enumerate(zip(top, tiles), 1):
            at = (0, r - 1) if maps is not None else divmod(r - 1, cols)
            axes = [fig.add_subplot(cells[at])]
            axes[0].imshow(img)
            axes[0].set_title(f"{r}   {contrib[j]:+.2f}% of evidence", fontsize=8, color=INK_2)
            if maps is not None:
                axes.append(fig.add_subplot(cells[1, r - 1]))
                axes[1].imshow(img)
                axes[1].imshow(maps[r - 1], cmap="magma", alpha=0.5, interpolation="bilinear",
                               extent=(0, img.size[0], img.size[1], 0))
            for a in axes:
                a.set_xticks([]), a.set_yticks([])
                for side in a.spines.values():
                    side.set_visible(False)
        label = "C  Top tiles (H&E)" + (";  D  below: where Virchow looked (attention)" if maps is not None else "")
    else:
        label = "C  Top tiles: add --slides with the slide file to show the real H&E tiles"
    fig.text(0.125, 0.43, label, color=INK, fontsize=11)

    verdict = "right" if info["pred"] == slide["tumour_type"] else f"wrong (called {info['pred']})"
    fig.suptitle(f"{slide['tumour_type']}  ·  slide {Path(str(slide['slide_id'])).stem}  ·  model {verdict}, "
                 f"p({info['target']}) = {info['prob_target']:.2f}  ·  {info['positive_share']:.0%} of tiles push towards it",
                 color=INK, fontsize=12, x=0.125, ha="left")
    return _save(fig, path)


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/explain")
    p.add_argument("--only", nargs="*", default=None, help="slide ids to explain (default: all)")
    p.add_argument("--slides", default=None, help="folder with the slide files, for real tiles (panel C)")
    p.add_argument("--download", action="store_true",
                   help="fetch missing slides from EBRAINS into --slides (needs EBRAINS_TOKEN), deleted after use")
    p.add_argument("--no-attention", action="store_true", help="skip Virchow attention (panel D)")
    p.add_argument("--target", default=None, help="type to explain (default: the slide's true type)")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    slides = load(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {Path(str(s)).stem for s in args.only} if args.only else None
    model = None
    rows = []
    for i, s in enumerate(slides):
        stem = Path(str(s["slide_id"])).stem
        if wanted and stem not in wanted:
            continue
        if s["coords"] is None:
            print(f"{stem}: no tile coordinates, skipped")
            continue
        contrib, info = tile_contributions(slides, i, args.target, args.seed)
        xy2d, method = embed_2d(s["features"], "umap", args.seed)
        top = top_tiles(contrib, args.top)

        import torch
        step = torch.load(Path(args.features) / f"{stem}.pt", weights_only=False).get("tile_step_level0")
        if step is None:  # older feature files: the tile grid spacing is the step
            gaps = np.diff(np.unique(s["coords"][:, 0]))
            step = int(gaps[gaps > 0].min())
        tiles = thumb = scale = maps = None
        path = find_slide(args.slides, s["slide_id"])
        fetched = False
        if path is None and args.download and args.slides:
            from brainpath import ebrains
            from brainpath.extract import download
            Path(args.slides).mkdir(parents=True, exist_ok=True)
            path = str(Path(args.slides) / f"{stem}.ndpi")
            print(f"{stem}: downloading the slide from EBRAINS")
            download(ebrains.signed_url(ebrains.object_name(s["tumour_type"], s["slide_id"])), path, auth=False)
            fetched = True
        if path:
            tiles, thumb, scale = read_tiles(path, s["coords"][top], step)
            if not args.no_attention:
                if model is None:
                    from brainpath.extract import load_virchow
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, transform = load_virchow(device)
                maps = attention_maps(model, transform, tiles, device)
            if fetched:
                Path(path).unlink()
        elif args.slides:
            print(f"{stem}: slide file not found in {args.slides}, panels C-D left out")

        plot_explanation(s, contrib, info, xy2d, tiles, thumb, scale, maps, step, args.top,
                         out / f"{stem}_explain.png")
        pd.DataFrame({"x": s["coords"][:, 0], "y": s["coords"][:, 1], "contribution": contrib}) \
            .to_csv(out / f"{stem}_tiles.csv", index=False)
        rows.append({"slide_id": s["slide_id"], "tumour_type": s["tumour_type"], **info,
                     "top_tiles_level0": s["coords"][top].tolist()})
        print(f"{stem}: {s['tumour_type']} -> {info['pred']}  ({out / f'{stem}_explain.png'})")
    (out / "explain_summary.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
