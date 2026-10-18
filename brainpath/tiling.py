"""Cut a whole-slide image into Virchow-sized tiles that contain tissue.

Virchow v1 expects 224 x 224 RGB tiles at 0.5 microns per pixel (20x).
EBRAINS slides are scanned at 40x (about 0.228 um/px), so each tile covers
roughly 491 x 491 level-0 pixels and is resized down to 224.
"""

import math

import numpy as np
from PIL import Image

TILE_PX = 224
TARGET_MPP = 0.5
DEFAULT_BASE_MPP = 0.2283  # Hamamatsu NanoZoomer 2.0 HT at 40x


def tile_grid(width, height, step):
    """Top-left level-0 coordinates of a non-overlapping grid of step x step tiles."""
    xs = np.arange(0, width - step + 1, step)
    ys = np.arange(0, height - step + 1, step)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.int64).reshape(-1, 2)


def tissue_mask(thumbnail):
    """Boolean tissue mask from an RGB thumbnail, thresholding saturation with Otsu."""
    hsv = np.asarray(thumbnail.convert("HSV"))
    sat = hsv[..., 1].astype(np.float64)
    return sat > otsu_threshold(sat)


def otsu_threshold(values):
    hist, edges = np.histogram(values, bins=256, range=(0, 256))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mean0 = m0 / np.maximum(w0, 1)
    mean1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mean0 - mean1) ** 2
    return centers[np.argmax(between)]


def tissue_fraction(mask, coords, step, mask_downsample):
    """Share of each tile's footprint that the mask marks as tissue."""
    s = max(1, int(round(step / mask_downsample)))
    out = np.empty(len(coords))
    for i, (x, y) in enumerate(coords):
        mx, my = int(x / mask_downsample), int(y / mask_downsample)
        out[i] = mask[my:my + s, mx:mx + s].mean() if mask[my:my + s, mx:mx + s].size else 0.0
    return out


class SlideTiler:
    """Finds tissue tiles in one slide and reads them as 224 x 224 PIL images."""

    def __init__(self, path, min_tissue=0.5, max_tiles=None, seed=0):
        import openslide  # imported here so the rest of the package works without it

        self.slide = openslide.OpenSlide(path)
        mpp = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        self.base_mpp = float(mpp) if mpp else DEFAULT_BASE_MPP
        self.step = int(round(TILE_PX * TARGET_MPP / self.base_mpp))

        width, height = self.slide.dimensions
        thumb_ds = 32
        thumb = self.slide.get_thumbnail((width // thumb_ds, height // thumb_ds)).convert("RGB")
        mask_ds = width / thumb.size[0]

        coords = tile_grid(width, height, self.step)
        keep = tissue_fraction(tissue_mask(thumb), coords, self.step, mask_ds) >= min_tissue
        coords = coords[keep]
        if max_tiles and len(coords) > max_tiles:
            idx = np.random.default_rng(seed).choice(len(coords), max_tiles, replace=False)
            coords = coords[np.sort(idx)]
        self.coords = coords

        self.level = self.slide.get_best_level_for_downsample(self.step / TILE_PX)
        self.level_ds = self.slide.level_downsamples[self.level]
        self.read_px = int(math.ceil(self.step / self.level_ds))

    def __len__(self):
        return len(self.coords)

    def read(self, i):
        x, y = self.coords[i]
        region = self.slide.read_region((int(x), int(y)), self.level, (self.read_px, self.read_px))
        return region.convert("RGB").resize((TILE_PX, TILE_PX), Image.BILINEAR)

    def close(self):
        self.slide.close()
