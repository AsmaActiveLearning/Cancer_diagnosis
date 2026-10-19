import json
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainpath import train
from brainpath.select_pilot import select_pilot
from brainpath.tiling import tile_grid, tissue_fraction, tissue_mask


def make_annotations():
    counts = {"GBM": 40, "Meningioma": 30, "Pilocytic": 25, "Medullo": 20, "Common5": 18,
              "RareA": 6, "RareB": 5, "RareC": 8, "RareD": 4, "Tiny1": 3, "Tiny2": 3, "Single": 1}
    rows = [{"uuid": f"{t}_{i}", "diagnosis": t} for t, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_select_pilot_roles_and_sizes():
    pilot = select_pilot(make_annotations(), "diagnosis", rare_max=10)
    sizes = pilot.groupby("role").diagnosis.nunique().to_dict()
    assert sizes == {"common": 4, "rare": 4, "heldout": 2}
    assert set(pilot[pilot.role == "common"].diagnosis) == {"GBM", "Meningioma", "Pilocytic", "Medullo"}
    assert (pilot[pilot.role == "common"].groupby("diagnosis").size() == 10).all()
    assert (pilot[pilot.role == "rare"].groupby("diagnosis").size() == 4).all()
    assert "Single" not in set(pilot.diagnosis)
    assert pilot.uuid.is_unique


def test_select_pilot_reports_too_few_rare_types():
    with pytest.raises(ValueError):
        select_pilot(make_annotations(), "diagnosis", n_rare=10, rare_max=10)


def test_tile_grid_and_tissue_fraction():
    coords = tile_grid(1000, 500, 491)
    assert coords.tolist() == [[0, 0], [491, 0]]
    mask = np.zeros((16, 32), dtype=bool)
    mask[:, :16] = True  # left half is tissue
    frac = tissue_fraction(mask, coords, 491, mask_downsample=31.25)
    assert frac[0] == pytest.approx(1.0) and frac[1] < 0.5


def test_tissue_mask_separates_stain_from_background():
    img = np.full((40, 40, 3), 240, dtype=np.uint8)  # near-white background
    img[:, :20] = (200, 80, 160)  # pink/purple tissue
    mask = tissue_mask(Image.fromarray(img))
    assert mask[:, :20].all() and not mask[:, 20:].any()


def synthetic_slides(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    spec = [("common", t, 6) for t in ("A", "B", "C")] + [("rare", "R1", 3), ("rare", "R2", 3),
                                                          ("heldout", "H", 4)]
    centers = {t: rng.normal(size=dim) * 3 for _, t, _ in spec}
    slides = []
    for role, t, n in spec:
        for i in range(n):
            feats = centers[t] + rng.normal(size=(50, dim))
            slides.append({"slide_id": f"{t}_{i}", "tumour_type": t, "role": role, "patient": f"{t}_{i}",
                           "features": torch.tensor(feats, dtype=torch.float32)})
    return slides


@pytest.mark.parametrize("method", ["mean", "abmil"])
def test_train_runs_and_separates_easy_classes(method):
    summary, preds, attention = train.run(synthetic_slides(), method=method, n_splits=3, epochs=15)
    assert summary["common_balanced_accuracy"] > 0.9
    assert summary["rare_accuracy"] > 0.9
    assert "unseen_auroc_prototype" in summary
    assert set(preds.role) == {"common", "rare", "heldout"}
    if method == "abmil":
        assert len(attention) == 18
        assert all(np.isclose(a.sum(), 1.0) for a in attention.values())


def test_load_features_reads_extract_output(tmp_path):
    torch.save({"features": torch.zeros(5, 8).half(), "coords": torch.zeros(5, 2).int(), "slide_id": "s1",
                "tumour_type": "A", "role": "common", "meta": {"patient_id": "p9"}}, tmp_path / "s1.pt")
    slides = train.load_features(tmp_path, patient_col="patient_id")
    assert slides[0]["patient"] == "p9" and slides[0]["features"].dtype == torch.float32


def test_extract_end_to_end_with_stand_in_model(tmp_path, monkeypatch):
    """Runs the extract loop on a small synthetic slide, with a stand-in for Virchow."""
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("openslide")
    from brainpath import extract

    img = np.full((2048, 2048, 3), 240, np.uint8)
    img[200:1800, 200:1200] = (200, 80, 160)
    slide = tmp_path / "s1.tif"
    res = 1e4 / 0.2283
    tifffile.imwrite(slide, img, tile=(256, 256), photometric="rgb", resolution=(res, res),
                     resolutionunit="CENTIMETER", compression="zlib")

    class StandIn(torch.nn.Module):  # same output shape as Virchow v1: (B, 257, 1280)
        def forward(self, x):
            return x.mean(dim=(1, 2, 3)).view(-1, 1, 1).expand(-1, 257, 1280)

    monkeypatch.setattr(extract, "load_virchow", lambda device: (StandIn(), lambda im: torch.zeros(3, 224, 224)))
    manifest = tmp_path / "pilot.csv"
    pd.DataFrame([{"slide_id": "s1.tif", "tumour_type": "A", "role": "common", "path": str(slide)}]).to_csv(manifest, index=False)
    monkeypatch.setattr("sys.argv", ["extract", str(manifest), "--out", str(tmp_path / "feats")])
    extract.main()

    d = torch.load(tmp_path / "feats" / "s1.pt", weights_only=False)
    assert d["features"].shape[1] == 2560 and len(d["features"]) == len(d["coords"]) > 0
    assert slide.exists()  # local files given via 'path' are never deleted


def test_openset_detector_scores_far_slides_higher():
    from brainpath.openset import SCORES, OpenSetDetector

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 32)) * 3
    X = np.concatenate([c + rng.normal(size=(8, 32)) for c in centers])
    y = np.repeat(["A", "B", "C"], 8)
    det = OpenSetDetector(pca_dim=8).fit(X, y)
    near = det.score(centers[0] + rng.normal(size=(5, 32)))
    far = det.score(rng.normal(size=(5, 32)) * 3 + 10)
    assert set(near) == set(SCORES)
    assert all(far[n].mean() > near[n].mean() for n in SCORES)


def test_openset_rotation_separates_unseen_types(tmp_path, monkeypatch):
    from brainpath import openset

    slides = synthetic_slides()
    assert openset.rotations(slides) == [["R1"], ["R2"], ["H"]]
    table, scores, per_rotation = openset.run(slides, n_splits=3)
    assert set(table.score) == set(openset.SCORES)
    assert (table.auroc_mean > 0.9).all()
    assert (table.known_flagged_mean <= 0.2).all()
    assert set(scores.hidden) == {"R1", "R2", "H"}
    assert all(r["known_accuracy"] > 0.9 for r in per_rotation)

    for s in slides:
        torch.save({"features": s["features"].half(), "slide_id": s["slide_id"], "tumour_type": s["tumour_type"],
                    "role": s["role"], "meta": {}}, tmp_path / f"{s['slide_id']}.pt")
    monkeypatch.setattr("sys.argv", ["openset", str(tmp_path), "--out", str(tmp_path / "out"), "--folds", "3"])
    openset.main()
    assert (tmp_path / "out" / "openset_summary.csv").exists()


def test_ebrains_object_name_matches_atlas_layout():
    from brainpath.ebrains import object_name

    assert (object_name("Glioblastoma, IDH-wildtype", "a194f783-357f-11eb-9ee2-001a7dda7111")
            == "v1.0/Glioblastoma, IDH-wildtype/a194f783-357f-11eb-9ee2-001a7dda7111.ndpi")
    assert object_name("X", "s.ndpi") == "v1.0/X/s.ndpi"


def test_extract_fetches_from_ebrains_without_url_column(tmp_path, monkeypatch):
    from brainpath import ebrains, extract

    calls = []
    monkeypatch.setattr(ebrains, "signed_url", lambda obj: calls.append(obj) or "https://signed/x")
    def fake_download(url, dest, auth=True):
        calls.append((url, auth))
        raise RuntimeError("stop after download")
    monkeypatch.setattr(extract, "download", fake_download)
    monkeypatch.setattr(extract, "load_virchow", lambda device: (None, None))
    manifest = tmp_path / "pilot.csv"
    pd.DataFrame([{"slide_id": "u1", "tumour_type": "Haemangioblastoma", "role": "heldout"}]).to_csv(manifest, index=False)
    monkeypatch.setattr("sys.argv", ["extract", str(manifest), "--out", str(tmp_path / "f"), "--tmp", str(tmp_path)])
    with pytest.raises(RuntimeError):
        extract.main()
    assert calls == ["v1.0/Haemangioblastoma/u1.ndpi", ("https://signed/x", False)]


def test_explore_runs_on_synthetic_features(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from brainpath import explore

    rng = np.random.default_rng(1)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for s in synthetic_slides(dim=24):
        n = len(s["features"])
        coords = np.stack(np.meshgrid(np.arange(10), np.arange(5)), -1).reshape(-1, 2)[:n] * 224
        torch.save({"features": s["features"].half(), "coords": torch.from_numpy(coords).int(),
                    "slide_id": s["slide_id"], "tumour_type": s["tumour_type"], "role": s["role"],
                    "meta": {"pat_id": s["patient"], "age": float(rng.integers(5, 80)),
                             "tissue_area": float(rng.uniform(5, 100))}}, feat_dir / f"{s['slide_id']}.pt")
    monkeypatch.setattr("sys.argv", ["explore", str(feat_dir), "--out", str(tmp_path / "out"), "--clusters", "4"])
    explore.main()

    out = tmp_path / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["slides"] == 28 and summary["known_1nn_accuracy"] > 0.9
    assert summary["nearest_known_sim_heldout"] < summary["nearest_known_sim_known"]
    for name in ("pca_variance.png", "map_pca.png", "similarity.png", "nearest_known.png", "tile_clusters.png"):
        assert (out / name).exists()
    assert len(list(out.glob("tilemap_*.png"))) == 6


def _write_synthetic_features(feat_dir, names=None):
    feat_dir.mkdir(parents=True, exist_ok=True)
    for s in synthetic_slides(dim=24):
        n = len(s["features"])
        coords = np.stack(np.meshgrid(np.arange(10), np.arange(5)), -1).reshape(-1, 2)[:n] * 224
        t = (names or {}).get(s["tumour_type"], s["tumour_type"])
        torch.save({"features": s["features"].half(), "coords": torch.from_numpy(coords).int(),
                    "slide_id": s["slide_id"], "tumour_type": t, "role": s["role"],
                    "meta": {"age": 54.0, "sex": "female", "location": "['frontal']", "laterality": "left"}},
                   feat_dir / f"{s['slide_id']}.pt")


def test_report_flags_heldout_and_uses_who_template(tmp_path, monkeypatch):
    from brainpath import report

    names = {"A": "Glioblastoma, IDH-wildtype", "B": "Pilocytic astrocytoma", "C": "Pituitary adenoma",
             "R1": "Immature teratoma", "R2": "Leiomyosarcoma", "H": "Haemangioblastoma"}
    _write_synthetic_features(tmp_path / "features", names)
    monkeypatch.setattr("sys.argv", ["report", str(tmp_path / "features"), "--out", str(tmp_path / "out"), "--min-prob", "0"])
    report.main()

    summary = pd.read_csv(tmp_path / "out" / "reports_summary.csv")
    assert len(summary) == 28
    known = summary[summary.role != "heldout"]
    assert known.top_correct.mean() > 0.9 and known.unknown_flag.mean() <= 0.1
    assert summary[summary.role == "heldout"].unknown_flag.all()
    gbm = (tmp_path / "out" / "A_0.md").read_text()
    for section in ("Integrated diagnosis", "CNS WHO grade", "Molecular information", "RESEARCH DRAFT"):
        assert section in gbm
    assert "Glioblastoma, NOS (IDH status not assessed)" in gbm and "CNS WHO grade 4" in gbm and "54 years, female" in gbm and "left, frontal" in gbm
    assert "not recognised" in (tmp_path / "out" / "H_0.md").read_text()


def test_report_llm_text_is_rejected_when_it_contradicts_the_facts():
    from brainpath import report

    f = {"integrated_diagnosis": "Hemangioblastoma"}
    good = '{"microscopy": "m", "comment": "Integrated diagnosis: Hemangioblastoma. ok"}'
    bad = '{"microscopy": "m", "comment": "Integrated diagnosis: Glioblastoma"}'
    assert report.llm_write(f, lambda p: good)[2] == "accepted"
    assert report.llm_write(f, lambda p: bad)[2].startswith("rejected")
    assert report.llm_write(f, lambda p: "no json here")[2].startswith("rejected")


def test_report_low_confidence_gives_differential_not_a_diagnosis():
    from brainpath import report

    row = {"slide_id": "s", "pred": "Glioblastoma, IDH-wildtype", "prob": 0.20, "unknown": False,
           "top3": [("Glioblastoma, IDH-wildtype", 0.20), ("Pilocytic astrocytoma", 0.20), ("Pituitary adenoma", 0.1)],
           "unseen_score": 0.1, "unseen_threshold": 0.2, "n_tiles": 10, "typical_tiles": [], "meta": {}}
    f = report.facts(row, min_prob=0.5)
    assert f["model_call"] == "indeterminate"
    assert "Glioblastoma, IDH-wildtype vs Pilocytic astrocytoma" in f["integrated_diagnosis"]
    md = report.render(f)
    assert "Indeterminate" in md and "WHO CNS5 entity if confirmed" not in md and "grade 4" not in md
    assert report.facts(row, min_prob=0.1)["model_call"] == "Glioblastoma, IDH-wildtype"
