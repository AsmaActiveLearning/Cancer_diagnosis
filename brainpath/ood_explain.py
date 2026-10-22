"""Show why some unseen tumour types are caught and others are not.

Each rare or held-out type is hidden in turn, as in openset.py, and scored with
the standardised prototype score (the one report.py uses). Hidden types are then
grouped by how well the score separates them from known slides (AUROC):
  far-OOD       AUROC >= 0.9   easy to detect: unlike every known type
  intermediate  0.7 - 0.9      partly detected
  near-OOD      AUROC < 0.7    hard: looks like a known type

Figures:
  ood_overview.png  every hidden type's slide scores against the known slides,
                    grouped by category, with the threshold
  <type>.png        one slide of that type (the median-scoring one, or every slide
                    with --all), with:
    A1  the slide (needs --slides)
    A2  tile novelty: how unlike every known type each tile is
    A3  the known type each tile is closest to (one colour per type)
    B   Virchow embedding space: the slide's tiles among tiles from known slides
    E   this slide's score among the known slides' scores, with the threshold
    C/D the most novel tiles and Virchow's attention in them (needs --slides)
    and a caption that interprets the figure from its numbers.

Tile novelty = 1 - cosine similarity to the nearest known-type centroid, in the
detector's standardised space. A tile counts as novel when it is more novel than
95% of tiles from known slides.

Examples:
  laptop:  python -m brainpath.ood_explain features/ --out results/ood/
  Kaggle:  python -m brainpath.ood_explain features/ --out results/ood/ --slides /kaggle/tmp --download
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brainpath.explore import INK, INK_2, SURFACE, _save, _style, load
from brainpath.openset import OpenSetDetector, evaluate_split, rotations

# Reference categorical palette, in slot order; categories reuse slots 3, 4 and 8.
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CATEGORIES = [("far-OOD", 0.9, "#1baf7a"), ("intermediate", 0.7, "#eda100"), ("near-OOD", -1, "#e34948")]
KNOWN_GREY = "#b5b3ad"


def category(auroc):
    for name, floor, color in CATEGORIES:
        if auroc >= floor:
            return name, color


def as_torch(slides):
    """openset.evaluate_split expects torch features and a patient id."""
    return [{**s, "features": torch.from_numpy(np.asarray(s["features"], dtype=np.float32)),
             "patient": str(s["meta"].get("pat_id", s["slide_id"]))} for s in slides]


def tile_view(det, features):
    """Per-tile novelty and nearest known type, in the detector's space."""
    sims = det.prep(features) @ det.centroids_.T
    return 1 - sims.max(1), sims.argmax(1)


def analyse_type(slides, hidden, seed=0, ref_tiles=200):
    rows, metrics = evaluate_split(as_torch(slides), [hidden], seed=seed, standardize=True)
    m = metrics["prototype"]
    known = [s for s in slides if s["tumour_type"] != hidden]
    det = OpenSetDetector(seed=seed, standardize=True).fit(
        np.stack([s["features"].mean(0) for s in known]), [s["tumour_type"] for s in known])
    rng = np.random.default_rng(seed)
    ref = np.concatenate([tile_view(det, s["features"][rng.choice(len(s["features"]), min(ref_tiles, len(s["features"])),
                                                                    replace=False)])[0] for s in known])
    name, color = category(m["auroc"])
    scores = pd.DataFrame(rows)[["slide_id", "tumour_type", "status", "prototype"]]
    return {"hidden": hidden, "auroc": m["auroc"], "threshold": m["threshold"], "detected": m["unseen_detected"],
            "known_flagged": m["known_flagged"], "category": name, "color": color, "det": det,
            "tile_threshold": float(np.quantile(ref, 0.95)), "scores": scores}


def interpret(res, slide, novel_share, nearest_share, score):
    flagged = score > res["threshold"]
    top = ", ".join(f"{t} ({p:.0%})" for t, p in nearest_share[:2])
    head = (f"{res['hidden']} hidden from training: {res['category']} (AUROC {res['auroc']:.2f}; "
            f"{res['detected']:.0%} of its slides flagged at a threshold that flags {res['known_flagged']:.0%} "
            f"of known slides). ")
    this = (f"This slide scores {score:.3f} against a threshold of {res['threshold']:.3f}, so it is "
            + ("flagged as an unknown type. " if flagged else "NOT flagged: the model takes it for a known type. "))
    tiles = f"{novel_share:.0%} of its tiles are more novel than 95% of tiles from known slides (A2). "
    if res["category"] == "far-OOD":
        why = (f"Its tiles lie outside the known clusters in Virchow space (B); the closest known types, {top}, "
               "are only partial matches. Its morphology is unlike anything the model was trained on, "
               "so the prototype distance separates it well.")
    elif res["category"] == "intermediate":
        why = (f"Part of the slide sits inside known clusters, mainly {top} (A3), and part outside. "
               "Detection depends on how much of the slide shows the distinctive pattern, so some slides "
               "of this type are caught and others are not.")
    else:
        why = (f"Its tiles fall inside the clusters of known types, mainly {top} (A3, B). To the model it is a "
               "variant of a known type, not a new one: morphology in H&E does not separate them, and "
               "molecular markers or finer-grained training types would be needed.")
    return head + this + tiles + why


# ----------------------------------------------------------------- figures

def plot_overview(results, path=None):
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: -r["auroc"])
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(results) + 2.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    _style(ax, "Unseen-type score per hidden type (standardised prototype)", "unseen score (higher = less like any known type)")
    rng = np.random.default_rng(0)
    for k, r in enumerate(results):
        sc = r["scores"]
        known, unseen = sc[sc.status == "known"].prototype, sc[sc.status == "unseen"].prototype
        ax.scatter(known, k + rng.uniform(-0.12, 0.12, len(known)), s=14, color=KNOWN_GREY, linewidths=0, zorder=2)
        ax.scatter(unseen, k + rng.uniform(-0.12, 0.12, len(unseen)), s=46, color=r["color"],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.plot([r["threshold"]] * 2, [k - 0.3, k + 0.3], color=INK, linewidth=1.2, zorder=4)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{r['hidden']}\n{r['category']}, AUROC {r['auroc']:.2f}" for r in results], fontsize=8)
    for t, r in zip(ax.get_yticklabels(), results):
        t.set_color(r["color"] if r["category"] != "intermediate" else "#9a6a00")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=KNOWN_GREY, label="known slides")]
    handles += [Line2D([], [], marker="o", linestyle="", color=c, label=f"hidden type: {n}") for n, _, c in CATEGORIES]
    handles += [Line2D([], [], color=INK, label="threshold (95% of known slides accepted)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    caption = ("Each row hides one tumour type from training and scores every slide. Far-OOD types (green) "
               "sit to the right of the threshold: their morphology is unlike any known type. Near-OOD types (red) "
               "overlap the known slides or sit left of them: they look like a known type and are not flagged.")
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(caption, 150)), color=INK_2, fontsize=9, va="top")
    return _save(fig, path)


def plot_slide(res, slide, novelty, nearest, xy2d, n_known_pts, known_types_2d, score, caption,
               tiles=None, thumb=None, thumb_scale=None, maps=None, step=None, k=6, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    det = res["det"]
    classes = list(det.classes_)
    colors = {t: TYPE_COLORS[i % len(TYPE_COLORS)] for i, t in enumerate(classes)}
    xy = slide["coords"]
    if step is None:
        gaps = np.diff(np.unique(xy[:, 0]))
        step = int(gaps[gaps > 0].min()) if (gaps > 0).any() else 1
    top = np.argsort(novelty)[::-1][:k]
    cmap = plt.get_cmap("magma_r")
    norm = Normalize(vmin=np.percentile(novelty, 2), vmax=max(np.percentile(novelty, 98), res["tile_threshold"]))

    tile_rows = 0.8 if tiles is not None and maps is not None else 0.45 if tiles is not None else 0.08
    fig = plt.figure(figsize=(18, 11 + 5 * tile_rows), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(4, 6, height_ratios=[1.2, 1.2, tile_rows, 0.4], hspace=0.38, wspace=0.35)

    def tile_map(ax, colours, title, alpha=1.0):
        _style(ax, title)
        ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
        if thumb is not None:
            px, s = xy / thumb_scale, step / thumb_scale
            x0, y0 = np.maximum(px.min(0) - 3 * s, 0)
            x1, y1 = np.minimum(px.max(0) + 4 * s, thumb.size)
            ax.imshow(thumb.crop((int(x0), int(y0), int(x1), int(y1))), alpha=0.55)
            pts = px - (int(x0), int(y0))
        else:
            pts, s = xy, step
        for (x, y), c in zip(pts, colours):
            ax.add_patch(Rectangle((x, y), s, s, color=c, alpha=alpha, linewidth=0))
        for r, j in enumerate(top, 1):
            ax.annotate(str(r), pts[j] + s / 2, color=INK, fontsize=7, ha="center", va="center", weight="bold")
        if thumb is None:
            ax.set_xlim(pts[:, 0].min() - s, pts[:, 0].max() + 2 * s)
            ax.set_ylim(pts[:, 1].max() + 2 * s, pts[:, 1].min() - s)
        ax.set_aspect("equal")

    # A1 the slide
    ax = fig.add_subplot(grid[0, 0:2])
    _style(ax, "A1  The slide (H&E)")
    ax.grid(False), ax.set_xticks([]), ax.set_yticks([])
    if thumb is not None:
        px, s = xy / thumb_scale, step / thumb_scale
        x0, y0 = np.maximum(px.min(0) - 3 * s, 0)
        x1, y1 = np.minimum(px.max(0) + 4 * s, thumb.size)
        ax.imshow(thumb.crop((int(x0), int(y0), int(x1), int(y1))))
        for r, j in enumerate(top, 1):
            p = px[j] - (int(x0), int(y0))
            ax.add_patch(Rectangle(p, s, s, fill=False, edgecolor=INK, linewidth=1.2))
    else:
        ax.text(0.5, 0.5, "The slide image needs the slide file:\nrun with --slides (Kaggle notebook)",
                ha="center", va="center", color=INK_2, fontsize=9, transform=ax.transAxes)

    # A2 novelty
    ax = fig.add_subplot(grid[0, 2:4])
    tile_map(ax, cmap(norm(novelty)), "A2  Tile novelty (dark = unlike every known type)", alpha=0.85)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax.inset_axes([1.03, 0.1, 0.03, 0.8]))
    cb.ax.axhline(res["tile_threshold"], color=INK, linewidth=1.2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.ax.set_title("novelty\n— known 95%", color=INK_2, fontsize=7)

    # A3 nearest known type
    ax = fig.add_subplot(grid[0, 4:6])
    tile_map(ax, [colors[classes[c]] for c in nearest], "A3  Closest known type per tile", alpha=0.8)

    # B embedding space
    ax = fig.add_subplot(grid[1, 0:4])
    _style(ax, "B  Virchow embedding space: this slide's tiles (dark) among tiles of known types (colour)")
    ax.set_xticks([]), ax.set_yticks([])
    kn, me = xy2d[:n_known_pts], xy2d[n_known_pts:]
    for t in classes:
        sel = known_types_2d == t
        ax.scatter(kn[sel, 0], kn[sel, 1], s=4, color=colors[t], alpha=0.35, linewidths=0)
        if sel.any():
            cx, cy = np.median(kn[sel], 0)
            ax.annotate(t if len(t) <= 26 else t[:24] + "…", (cx, cy), fontsize=7, color=INK, ha="center",
                        weight="bold", zorder=5)
    order = np.argsort(novelty)
    ax.scatter(me[order, 0], me[order, 1], s=7, c=cmap(norm(novelty[order])), linewidths=0, zorder=3)
    for r, j in enumerate(top, 1):
        ax.annotate(str(r), me[j], xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK, weight="bold")
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[t], label=t) for t in classes]
    handles.append(Line2D([], [], marker="o", linestyle="", color=cmap(0.85), label=f"this slide ({slide['tumour_type']})"))
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    # E slide score among known slides
    ax = fig.add_subplot(grid[1, 5])
    _style(ax, "E  Slide unseen score")
    sc = res["scores"]
    rng = np.random.default_rng(0)
    known = sc[sc.status == "known"].prototype
    ax.scatter(rng.uniform(-0.15, 0.15, len(known)), known, s=12, color=KNOWN_GREY, linewidths=0)
    others = sc[(sc.status == "unseen") & (sc.slide_id != slide["slide_id"])].prototype
    ax.scatter(rng.uniform(0.85, 1.15, len(others)), others, s=24, color=res["color"], alpha=0.5, linewidths=0)
    ax.scatter([1], [score], s=70, color=res["color"], edgecolor=INK, linewidth=1.2, zorder=4)
    ax.axhline(res["threshold"], color=INK, linewidth=1)
    ax.set_xticks([0, 1]), ax.set_xticklabels(["known", "hidden type"], fontsize=8)
    ax.set_xlim(-0.5, 1.5)

    # C/D tiles
    if tiles is not None:
        cells = grid[2, :].subgridspec(2 if maps is not None else 1, len(tiles), hspace=0.1, wspace=0.06)
        for r, (j, img) in enumerate(zip(top, tiles), 1):
            ax = fig.add_subplot(cells[0, r - 1])
            ax.imshow(img)
            ax.set_title(f"{r}  novelty {novelty[j]:.2f}, near {classes[nearest[j]][:18]}", fontsize=7, color=INK_2)
            axes = [ax]
            if maps is not None:
                ax2 = fig.add_subplot(cells[1, r - 1])
                ax2.imshow(img)
                ax2.imshow(maps[r - 1], cmap="magma", alpha=0.5, interpolation="bilinear",
                           extent=(0, img.size[0], img.size[1], 0))
                axes.append(ax2)
            for a in axes:
                a.set_xticks([]), a.set_yticks([])
                for side in a.spines.values():
                    side.set_visible(False)
        fig.text(0.125, grid[2, 0].get_position(fig).y1 + 0.025,
                 "C  Most novel tiles (H&E)" + ("; D below: where Virchow looked" if maps is not None else ""),
                 color=INK, fontsize=11)
    else:
        fig.text(0.125, grid[2, 0].get_position(fig).y1, "C  Most novel tiles: needs the slide file (--slides)",
                 color=INK_2, fontsize=10)

    # caption
    ax = fig.add_subplot(grid[3, :])
    ax.axis("off")
    ax.text(0, 1, "\n".join(textwrap.wrap(caption, 190)), color=INK, fontsize=10, va="top", transform=ax.transAxes)

    fig.suptitle(f"{res['hidden']}  ·  {res['category']}  ·  slide {Path(str(slide['slide_id'])).stem}",
                 color=res["color"] if res["category"] != "intermediate" else "#9a6a00", fontsize=14,
                 x=0.125, ha="left", weight="bold")
    return _save(fig, path)


def embed(det, slide, known, per_slide, seed):
    from brainpath.explore import embed_2d
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for s in known:
        idx = rng.choice(len(s["features"]), min(per_slide, len(s["features"])), replace=False)
        parts.append(det.prep(s["features"][idx]))
        labels += [s["tumour_type"]] * len(idx)
    Kz = np.concatenate(parts)
    xy, _ = embed_2d(np.concatenate([Kz, det.prep(slide["features"])]), "umap", seed)
    return xy, len(Kz), np.array(labels)


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/ood")
    p.add_argument("--rotate", choices=["rare+heldout", "all", "heldout"], default="rare+heldout")
    p.add_argument("--all", action="store_true", help="one figure per hidden slide (default: the median one per type)")
    p.add_argument("--slides", default=None, help="folder with slide files, for the H&E panels")
    p.add_argument("--download", action="store_true", help="fetch missing slides from EBRAINS into --slides")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--cache", default=None, help="where H&E tiles and attention are kept between runs "
                   "(default: <out>/cache)")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and read the slides again")
    p.add_argument("--known-tiles", type=int, default=80, help="tiles per known slide in the embedding plot")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from brainpath.explain import attention_maps, find_slide, load_cached, read_tiles, save_cached

    slides = load(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out / "cache"
    results = [analyse_type(slides, hidden, args.seed) for (hidden,) in rotations(slides, args.rotate)]
    plot_overview(results, out / "ood_overview.png")  # first, so a failed download cannot lose it
    print(f"wrote {out / 'ood_overview.png'}")
    summary = []
    model = None
    for res in results:
        hidden = res["hidden"]
        sc = res["scores"]
        mine = sc[sc.status == "unseen"].sort_values("prototype")
        chosen = mine.slide_id.tolist() if args.all else [mine.slide_id.iloc[len(mine) // 2]]
        known = [s for s in slides if s["tumour_type"] != hidden]
        print(f"{hidden}: {res['category']}, AUROC {res['auroc']:.2f}")
        for sid in chosen:
            s = next(x for x in slides if x["slide_id"] == sid)
            if s["coords"] is None:
                continue
            stem = Path(str(sid)).stem
            novelty, nearest = tile_view(res["det"], s["features"])
            classes = list(res["det"].classes_)
            counts = pd.Series([classes[c] for c in nearest]).value_counts(normalize=True)
            score = float(mine.set_index("slide_id").prototype[sid])
            novel_share = float((novelty > res["tile_threshold"]).mean())
            caption = interpret(res, s, novel_share, list(counts.items()), score)
            xy2d, n_known, known_labels = embed(res["det"], s, known, args.known_tiles, args.seed)

            step = torch.load(Path(args.features) / f"{stem}.pt", weights_only=False).get("tile_step_level0")
            if step is None:
                gaps = np.diff(np.unique(s["coords"][:, 0]))
                step = int(gaps[gaps > 0].min())
            tiles = thumb = scale = maps = None
            top = np.argsort(novelty)[::-1][:6]
            cached = None if args.refresh else load_cached(cache, stem, "ood", s["coords"][top],
                                                           not args.no_attention)
            path = None if cached else find_slide(args.slides, sid)
            fetched = False
            try:
                if cached:
                    print(f"  {stem}: H&E tiles and attention from the cache, no download")
                    tiles, thumb, scale, maps = cached
                elif path is None and args.download and args.slides:
                    from brainpath import ebrains
                    from brainpath.extract import download
                    Path(args.slides).mkdir(parents=True, exist_ok=True)
                    path = str(Path(args.slides) / f"{stem}.ndpi")
                    print(f"  {stem}: downloading the slide from EBRAINS")
                    fetched = True
                    download(ebrains.signed_url(ebrains.object_name(s["tumour_type"], sid)), path, auth=False)
                if path:
                    tiles, thumb, scale = read_tiles(path, s["coords"][top], step)
                    if not args.no_attention:
                        if model is None:
                            from brainpath.extract import load_virchow
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model, transform = load_virchow(device)
                        maps = attention_maps(model, transform, tiles, device)
                    save_cached(cache, stem, "ood", s["coords"][top], tiles, thumb, scale, maps)
            except Exception as e:  # keep going: the figure is drawn without the H&E panels
                print(f"  {stem}: could not read the slide ({type(e).__name__}: {e}); drawing without H&E panels")
                tiles = thumb = scale = maps = None
            finally:
                if fetched and path and Path(path).exists():
                    Path(path).unlink()

            name = f"{res['category']}_{hidden}_{stem}".replace(" ", "_").replace(",", "")
            plot_slide(res, s, novelty, nearest, xy2d, n_known, known_labels, score, caption,
                       tiles, thumb, scale, maps, step, path=out / f"{name}.png")
            summary.append({"hidden": hidden, "category": res["category"], "auroc": res["auroc"],
                            "slide_id": sid, "score": score, "threshold": res["threshold"],
                            "flagged": score > res["threshold"], "novel_tile_share": novel_share,
                            "closest_known": dict(counts.round(3)), "caption": caption})
            print(f"  {stem}: {'flagged' if score > res['threshold'] else 'not flagged'} -> {name}.png")
            (out / "ood_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "ood_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
