"""Generate a PRISM draft report for each slide from its saved Virchow v1 features.

PRISM (paige-ai/Prism) is the report baseline. It is gated on Hugging Face,
licensed for non-commercial research only, and its reports may contain errors:
they are research drafts for pathologist review, never diagnoses.

PRISM was trained on all tissue tiles of a slide. If features were extracted
with --max-tiles, reports come from a sample of tiles; note this in results.

Example:
  python -m brainpath.prism_report features/ --out results/prism_reports.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features")
    p.add_argument("--out", default="results/prism_reports.csv")
    p.add_argument("--num-beams", type=int, default=5)
    args = p.parse_args()

    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True).to(device).eval()
    dtype = next(model.parameters()).dtype

    rows = []
    for f in sorted(Path(args.features).glob("*.pt")):
        d = torch.load(f, weights_only=False)
        tiles = d["features"].unsqueeze(0).to(device, dtype)  # (1, n_tiles, 2560)
        reprs = model.slide_representations(tiles)
        ids = model.generate(key_value_states=reprs["image_latents"], do_sample=False,
                             num_beams=args.num_beams, num_beam_groups=1)
        report = model.untokenize(ids)
        report = report[0] if isinstance(report, (list, tuple)) else report
        rows.append({"slide_id": d["slide_id"], "tumour_type": d["tumour_type"], "role": d["role"],
                     "n_tiles": tiles.shape[1], "prism_report": report})
        print(f"{d['slide_id']} ({d['tumour_type']}): {report}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
