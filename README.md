# Rare and unseen brain tumour recognition with Virchow and PRISM

Pilot code for the project *Long-Tail and Open-Set Recognition of Brain Tumours with Evidence-Grounded Report Generation Using Pathology Foundation Models*.

It uses the [EBRAINS Digital Brain Tumour Atlas](https://doi.org/10.25493/WQ48-ZGX) (3,115 H&E slides, 126 tumour types), the [Virchow v1](https://huggingface.co/paige-ai/Virchow) pathology foundation model as a frozen feature extractor, and [PRISM](https://huggingface.co/paige-ai/Prism) as the report-generation baseline.

> Research use only. Virchow and PRISM forbid clinical diagnosis. PRISM and the EBRAINS data are non-commercial.

## How the work is split

| Step | Where | Script |
| --- | --- | --- |
| 1. Pick the pilot slides from the annotation CSV | Laptop | `brainpath.select_pilot` |
| 2. Download each slide, extract Virchow features, delete the slide | Kaggle or Colab (GPU) | `brainpath.extract`, `notebooks/kaggle_extract.ipynb` |
| 3. Draft PRISM reports from the features | Kaggle or Colab | `brainpath.prism_report` |
| 4. Train and evaluate: common, rare and unseen types | Laptop (CPU is fine) | `brainpath.train` |

Only one slide (about 1.2 GB on average) is on disk at a time during step 2. The saved features for a 4,000-tile slide are about 20 MB, so the whole pilot's features fit on a laptop.

## Pilot design

| Role | Types | Slides per type | Used for |
| --- | --- | --- | --- |
| common | 4 most frequent | 10 | Classifier training and cross-validated testing |
| rare | 4 with 4 to 15 slides | 4 | Few-shot recognition with prototypes |
| heldout | 2 others | 3 | Never trained on; tests unseen-type detection |

That is about 62 slides, or roughly 75 GB downloaded one at a time. Every number is a command-line option.

## Before you start

1. **Hugging Face:** request access to [Virchow](https://huggingface.co/paige-ai/Virchow) and [PRISM](https://huggingface.co/paige-ai/Prism) with your institutional email, then create a read token.
2. **EBRAINS:** register, accept the data use agreement, and download the annotation CSV for the atlas.

## 1. Select the pilot slides (laptop)

```bash
pip install -r requirements-laptop.txt
python -m brainpath.select_pilot annotations.csv --slide-col <slide column> --type-col <diagnosis column>
```

If you get a column name wrong, the script lists the available columns. Add `--exclude "<control type name>"` to skip control tissue. The output is `pilot_slides.csv`.

Then add one column to `pilot_slides.csv` so the extractor can find each slide:

- `url`: a direct download link per slide from EBRAINS. If the links need a login token, set it as `EBRAINS_TOKEN`.
- or `path`: a slide file you already downloaded.

## 2. Extract features (Kaggle)

Open `notebooks/kaggle_extract.ipynb` on Kaggle with a GPU accelerator. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`, and upload `pilot_slides.csv` as a Kaggle dataset. The notebook runs:

```bash
python -m brainpath.extract pilot_slides.csv --out features/ --max-tiles 4000
python -m brainpath.prism_report features/ --out results/prism_reports.csv
```

The extractor skips slides that are already done, so you can stop and restart it. Download the `features/` folder at the end.

## 3. Train and evaluate (laptop)

```bash
python -m brainpath.train features/ --out results/
```

This compares a mean-pooling baseline with attention-based MIL and writes:

- `summary.json`: balanced accuracy on common types, accuracy on rare types, and AUROC for detecting unseen types (from classifier confidence and from prototype similarity)
- `predictions_*.csv`: one row per slide
- `attention_abmil.npz`: attention weight per tile, the input to the explanation step

With about 60 slides, treat the numbers as a feasibility check, not a result.

### Unseen-type detection

```bash
python -m brainpath.openset features/ --out results/openset/
```

This hides each rare and held-out type in turn, fits a detector on the remaining (known) types, and compares five unseen scores: classifier confidence, energy, Mahalanobis distance, nearest-neighbour distance and prototype similarity. It writes:

- `openset_summary.csv`: one row per score with AUROC, false alarms at 95% detection, and the share of unseen slides caught at a threshold set from known slides only (mean and spread over rotations)
- `openset_scores.csv`: every slide's scores in every rotation
- `openset_rotations.json`: the full metrics per rotation

Use `--rotate all` to also hide each common type once, or `--rotate heldout` for the single held-out split.

## Tests

```bash
pip install -r requirements-laptop.txt openslide-python openslide-bin tifffile
python -m pytest
```

The tests use synthetic data and a stand-in for Virchow, so they need no downloads or GPU.

## Layout

```
brainpath/
  select_pilot.py   pick common, rare and held-out slides
  tiling.py         tissue detection and 224 px tiles at 0.5 um/px (20x)
  extract.py        download, Virchow v1 features, delete
  prism_report.py   PRISM draft reports
  train.py          mean-pool and ABMIL classifiers, prototypes, unseen-type scores
  openset.py        unseen-type detection: five scores, rotating hidden types, thresholds
notebooks/
  kaggle_extract.ipynb
tests/
```
