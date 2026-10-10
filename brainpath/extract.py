"""Download each pilot slide, extract Virchow v1 tile features, save them, delete the slide.

Runs on a GPU machine (Kaggle or Colab). Only one slide is on disk at a time,
and each saved feature file is a few tens of MB.

The manifest is pilot_slides.csv from select_pilot, plus either:
  url  - a download link for the slide (sent with EBRAINS_TOKEN as a bearer token if set), or
  path - a slide file already on disk (kept, not deleted).

Example:
  python -m brainpath.extract pilot_slides.csv --out features/ --max-tiles 4000
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from brainpath.tiling import SlideTiler

VIRCHOW_ID = "hf-hub:paige-ai/Virchow"


def load_virchow(device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(VIRCHOW_ID, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def virchow_embedding(output):
    """Class token concatenated with the mean patch token: (B, 257, 1280) -> (B, 2560)."""
    return torch.cat([output[:, 0], output[:, 1:].mean(1)], dim=-1)


@torch.inference_mode()
def embed_slide(tiler, model, transform, device, batch_size=64):
    feats = []
    for start in range(0, len(tiler), batch_size):
        batch = torch.stack([transform(tiler.read(i)) for i in range(start, min(start + batch_size, len(tiler)))])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch.to(device))
        feats.append(virchow_embedding(out).float().cpu())
    return torch.cat(feats).half()


def download(url, dest):
    import requests

    headers = {}
    if os.environ.get("EBRAINS_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['EBRAINS_TOKEN']}"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out", default="features")
    p.add_argument("--max-tiles", type=int, default=4000, help="random tissue tiles per slide (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tmp", default=tempfile.gettempdir(), help="where downloaded slides wait to be processed")
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "url" not in manifest.columns and "path" not in manifest.columns:
        raise SystemExit("manifest needs a 'url' or a 'path' column")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("warning: no GPU found, extraction will be very slow")
    model, transform = load_virchow(device)

    for n, row in enumerate(manifest.itertuples(index=False), 1):
        slide_id = str(row.slide_id)
        target = out_dir / f"{Path(slide_id).stem}.pt"
        if target.exists():
            print(f"[{n}/{len(manifest)}] {slide_id}: already done")
            continue

        local = getattr(row, "path", None)
        downloaded = not (isinstance(local, str) and Path(local).exists())
        if downloaded:
            local = str(Path(args.tmp) / Path(slide_id).name)
            if not local.endswith(".ndpi"):
                local += ".ndpi"
            t0 = time.time()
            download(row.url, local)
            print(f"[{n}/{len(manifest)}] {slide_id}: downloaded in {time.time() - t0:.0f}s")

        try:
            t0 = time.time()
            tiler = SlideTiler(local, max_tiles=args.max_tiles or None)
            features = embed_slide(tiler, model, transform, device, args.batch_size)
            torch.save({
                "features": features,
                "coords": torch.from_numpy(tiler.coords).int(),
                "tile_step_level0": tiler.step,
                "slide_id": slide_id,
                "tumour_type": row.tumour_type,
                "role": row.role,
                "meta": {k: v for k, v in row._asdict().items() if k not in ("url", "path")},
            }, target)
            tiler.close()
            print(f"[{n}/{len(manifest)}] {slide_id}: {len(features)} tiles in {time.time() - t0:.0f}s")
        finally:
            if downloaded and os.path.exists(local):
                os.remove(local)


if __name__ == "__main__":
    main()
