import sys
import time
from pathlib import Path

import numpy as np
from skimage.io import imread

import hw2
from hw2 import (WEIGHT_SCHEMES, sample_Z, recover_g, merge_stack,
                 photographic_tonemap_rgb, photographic_tonemap_luminance)
from cp_hw2 import writeHDR, readHDR
from display_fn import save_tonemap, make_grid, single_vs_hdr


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "my_stack"
FAST = "--fast" in sys.argv
OUT_DIR = Path(__file__).resolve().parent / "deliverables" / "q4"

NUM_EXPOSURES = 10
LAMBDA = 50.0
STRIDE = 200  # for g recovery only

ZMIN_PER_SET = {"raw": 0.005, "jpg": 0.05}

MERGE_STRIDE = 1
PREVIEW_STRIDE = 4  # downsample for the comparison grids only

# the two .HDR deliverables: (mode, scheme)
RAW_PICK = ("log", "uniform")
JPG_PICK = ("log", "uniform")

# tonemap sweeps
K_VALUES = [0.09, 0.15, 0.36, 0.72]
B_FIXED = 0.95
B_VALUES = [0.7, 0.85, 0.95, 1.0]
K_FIXED = 0.15

# TODO: pick vals for final tonemap (source, method, K, B) 
FINAL = ("jpg", "luminance", 0.15, 0.95)


TONEMAPS = {
    "rgb": photographic_tonemap_rgb,
    "luminance": photographic_tonemap_luminance,
}


def merge_pick(src, paths, t, mode, scheme, g=None):

    out_path = OUT_DIR / f"my_hdr_{src}.hdr"
    if out_path.exists():
        print(f"{out_path.name} exists, skipping merge")
        return readHDR(str(out_path))

    hw2.ZMIN = ZMIN_PER_SET[src]  # weight fns + zero-weight repair read this global
    start = time.time()
    hdr = merge_stack(paths, t, WEIGHT_SCHEMES[scheme], is_photon=(scheme == "photon"),
                      g=g, mode=mode, stride=MERGE_STRIDE)
    writeHDR(str(out_path), hdr)
    print(f"{out_path.name} ({src}/{mode}/{scheme})  ({time.time()-start:.1f}s)  "
          f"min={hdr.min():.4g} max={hdr.max():.4g} mean={hdr.mean():.4g}")
    return hdr


def sweep_rows(hdrs, sweep, fixed):
    row_fns, row_labels = [], []
    for src, hdr in hdrs.items():
        for method, tm_fn in TONEMAPS.items():
            if sweep == "K":
                row_fns.append(lambda K, h=hdr, f=tm_fn: f(h, K=K, B=fixed))
            else:
                row_fns.append(lambda B, h=hdr, f=tm_fn: f(h, K=fixed, B=B))
            row_labels.append(f"{src} / {method}")
    return row_fns, row_labels


if __name__ == "__main__":
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t = np.array([2.0 ** (k - NUM_EXPOSURES) for k in range(1, NUM_EXPOSURES + 1)])
    jpg_paths = [DATA_DIR / f"exposure{k}.jpg" for k in range(1, NUM_EXPOSURES + 1)]
    tiff_paths = [DATA_DIR / f"exposure{k}.tiff" for k in range(1, NUM_EXPOSURES + 1)]

    # 1. recover g from my own jpgs 
    print(f"solving g(z) for the .jpg stack ({JPG_PICK[1]} weights)...")
    jpg_images = [imread(p) for p in jpg_paths]
    Z = sample_Z(jpg_images, stride=STRIDE)
    del jpg_images

    hw2.ZMIN = ZMIN_PER_SET["jpg"]
    jpg_mode, jpg_scheme = JPG_PICK
    g, _ = recover_g(Z, t, LAMBDA, WEIGHT_SCHEMES[jpg_scheme], is_photon=(jpg_scheme == "photon"))

    # 2. merge only the picked variants (RAW is already linear, so no g)
    hdrs_full = {
        "raw": merge_pick("raw", tiff_paths, t, *RAW_PICK),
        "jpg": merge_pick("jpg", jpg_paths, t, jpg_mode, jpg_scheme, g=g),
    }

    # 3. tonemap sweeps
    hdrs_small = {src: h[::PREVIEW_STRIDE, ::PREVIEW_STRIDE] for src, h in hdrs_full.items()}
    grid_size = (4 * len(K_VALUES), 3 * 2 * len(TONEMAPS))
    make_grid(*sweep_rows(hdrs_small, "K", B_FIXED), K_VALUES,
              lambda K: f"K={K}, B={B_FIXED}", OUT_DIR / "tonemap_K_sweep.png", figsize=grid_size, dpi=120)
    make_grid(*sweep_rows(hdrs_small, "B", K_FIXED), B_VALUES,
              lambda B: f"K={K_FIXED}, B={B}", OUT_DIR / "tonemap_B_sweep.png", figsize=grid_size, dpi=120)

    # 4. final tonemap at full res
    src, method, K, B = FINAL
    final_path = OUT_DIR / f"final_{src}_{method}_K{K}_B{B}.png"
    final_tm = save_tonemap(TONEMAPS[method](hdrs_full[src], K=K, B=B), final_path)
    picks = [0, NUM_EXPOSURES // 2, NUM_EXPOSURES - 1]  # shortest, middle, longest
    single_vs_hdr(jpg_paths, picks, final_tm, OUT_DIR / "single_vs_hdr.png", stride=PREVIEW_STRIDE)

    print(f"\ndone in {time.time()-t0:.1f}s, outputs in {OUT_DIR}")
