"""Pick the pilot slides from the EBRAINS annotation CSV.

Tumour types are split into three roles:
  common  - the most frequent types, used to train and test the classifier
  rare    - types with few slides, used for few-shot (prototype) recognition
  heldout - types never shown to the model, used to test unseen-type detection

Example:
  python -m brainpath.select_pilot annotations.csv --slide-col uuid --type-col diagnosis
"""

import argparse

import pandas as pd

GB_PER_SLIDE = 1.2  # rough average: 3.6 TB / 3,115 slides


def select_pilot(df, type_col, n_common=4, k_common=10, n_rare=4, k_rare=4,
                 n_heldout=2, k_heldout=3, rare_max=15, exclude=(), seed=0):
    """Return the selected rows of df with an added 'role' column."""
    df = df[~df[type_col].isin(exclude)]
    counts = df[type_col].value_counts()

    common = list(counts.index[:n_common])
    remaining = counts.drop(common)

    rare_pool = remaining[(remaining >= k_rare) & (remaining <= rare_max)]
    if len(rare_pool) < n_rare:
        raise ValueError(f"only {len(rare_pool)} types have {k_rare}-{rare_max} slides; "
                         f"lower --k-rare or raise --rare-max")
    rare = list(rare_pool.sample(n=n_rare, random_state=seed).index)

    heldout_pool = remaining.drop(rare)
    heldout_pool = heldout_pool[heldout_pool >= k_heldout]
    if len(heldout_pool) < n_heldout:
        raise ValueError(f"only {len(heldout_pool)} types left with at least {k_heldout} slides")
    heldout = list(heldout_pool.sample(n=n_heldout, random_state=seed).index)

    parts = []
    for role, types, k in (("common", common, k_common), ("rare", rare, k_rare),
                           ("heldout", heldout, k_heldout)):
        for t in types:
            rows = df[df[type_col] == t]
            parts.append(rows.sample(n=min(k, len(rows)), random_state=seed).assign(role=role))
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("annotations", help="EBRAINS annotation CSV")
    p.add_argument("--slide-col", required=True, help="column holding the slide file id or name")
    p.add_argument("--type-col", required=True, help="column holding the diagnosis / tumour type")
    p.add_argument("--out", default="pilot_slides.csv")
    p.add_argument("--n-common", type=int, default=4)
    p.add_argument("--k-common", type=int, default=10)
    p.add_argument("--n-rare", type=int, default=4)
    p.add_argument("--k-rare", type=int, default=4)
    p.add_argument("--n-heldout", type=int, default=2)
    p.add_argument("--k-heldout", type=int, default=3)
    p.add_argument("--rare-max", type=int, default=15, help="most slides a type may have to count as rare")
    p.add_argument("--exclude", nargs="*", default=[], help="types to skip, e.g. control tissue")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.annotations)
    for col in (args.slide_col, args.type_col):
        if col not in df.columns:
            raise SystemExit(f"column '{col}' not found; available columns: {list(df.columns)}")

    pilot = select_pilot(df, args.type_col, args.n_common, args.k_common, args.n_rare, args.k_rare,
                         args.n_heldout, args.k_heldout, args.rare_max, args.exclude, args.seed)
    pilot = pilot.rename(columns={args.slide_col: "slide_id", args.type_col: "tumour_type"})
    pilot.to_csv(args.out, index=False)

    print(pilot.groupby(["role", "tumour_type"]).size().to_string())
    print(f"\n{len(pilot)} slides, roughly {len(pilot) * GB_PER_SLIDE:.0f} GB to download -> {args.out}")


if __name__ == "__main__":
    main()
