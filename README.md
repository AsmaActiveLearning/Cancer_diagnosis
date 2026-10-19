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
python -m brainpath.select_pilot annotations.csv --slide-col uuid --type-col diagnosis --k-common 5
```

If you get a column name wrong, the script lists the available columns. Control slides have no diagnosis, so they are skipped automatically. The output is `pilot_slides.csv`.

The extractor finds each slide on EBRAINS from its `uuid` and diagnosis (the atlas stores them as `v1.0/<diagnosis>/<uuid>.ndpi`), so no download links are needed. It logs in with your EBRAINS access token in `EBRAINS_TOKEN`. Test the token on one slide with `python -m brainpath.ebrains pilot_slides.csv`.

To use other sources instead, add a `url` column (a direct download link) or a `path` column (a slide file already on disk).

## 2. Extract features (Kaggle)

Open `notebooks/kaggle_extract.ipynb` on Kaggle with a GPU accelerator. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`, and upload `pilot_slides.csv` as a Kaggle dataset. The notebook runs:

```bash
python -m brainpath.extract pilot_slides.csv --out features/ --max-tiles 4000
python -m brainpath.prism_report features/ --out results/prism_reports.csv
```

The extractor skips slides that are already done, so you can stop and restart it. Download the `features/` folder at the end.

## 3. Explore the embeddings (laptop)

Before training, look at what Virchow already captures. Unzip the Kaggle outputs into the repo folder, so the `.pt` files sit in `features/`, then open `notebooks/explore_embeddings.ipynb` in VS Code and run it cell by cell. It shows:

- PCA explained variance, and PCA and UMAP maps of the slides
- cosine similarity between slides, ordered by type
- separability: silhouette, nearest-neighbour accuracy, and how far held-out slides sit from known ones
- confounders: whether the main directions follow tile count, tissue area or age
- tile clusters shared across types, and a tile map per slide

To save every figure and table to a folder without the notebook, run `python -m brainpath.explore features/ --out results/explore/`.

## 4. Train and evaluate (laptop)

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

## 5. Draft neuropathology reports (laptop)

```bash
python -m brainpath.report features/ --out results/reports/
```

This writes one Markdown and one JSON file per slide. Each report follows the WHO CNS5 layered format from the ICCR dataset: integrated diagnosis, histopathological classification, CNS WHO grade, molecular information, microscopic findings and comment. Every field comes from facts the pipeline can show:
- the tumour type and its probability, from a classifier that never saw the slide
- an "unknown type" flag for slides far from every known type
- the most typical tiles, with their coordinates on the slide
- age, sex and location from the annotation file

What H&E cannot show (IDH, fusions, lineage) is never stated as a finding. The integrated diagnosis is given as NOS, with the tests to request.

The WHO CNS5 lookup table in `brainpath/report.py` should be checked by a neuropathologist. With `--llm google/medgemma-27b-text-it` (or any instruction-tuned model on a GPU), an LLM rewrites the findings and comment from the same facts. Its text is kept only if it repeats the integrated diagnosis correctly.

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
  ebrains.py        find and fetch atlas slides on the EBRAINS data proxy
  extract.py        download, Virchow v1 features, delete
  prism_report.py   PRISM draft reports
  explore.py        PCA, UMAP, similarity, separability and tile clusters
  report.py         WHO CNS5 layered draft reports grounded in the classifier
  train.py          mean-pool and ABMIL classifiers, prototypes, unseen-type scores
  openset.py        unseen-type detection: five scores, rotating hidden types, thresholds
notebooks/
  kaggle_extract.ipynb
  explore_embeddings.ipynb
tests/
```
