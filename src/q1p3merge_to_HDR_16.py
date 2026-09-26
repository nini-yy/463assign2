import time
from pathlib import Path

import numpy as np
from skimage.io import imread

from hw2 import DATA_DIR, NUM_EXPOSURES, WEIGHT_SCHEMES, sample_Z, recover_g, merge_stack
from cp_hw2 import writeHDR

OUT_DIR = Path(__file__).resolve().parent / "deliverables" / "q1" / "hdr_all16"
LAMBDA = 50.0
STRIDE = 200  # for g recovery only

if __name__ == "__main__":
    t = np.array([(1.0 / 2048.0) * 2.0 ** (k - 1) for k in range(1, NUM_EXPOSURES + 1)])

    jpg_paths = [DATA_DIR / f"exposure{k}.jpg" for k in range(1, NUM_EXPOSURES + 1)]
    tiff_paths = [DATA_DIR / f"exposure{k}.tiff" for k in range(1, NUM_EXPOSURES + 1)]

    # g depends on the weighting scheme 
    print("solving g(z) per weighting scheme for the .jpg stack...")
    jpg_images = [imread(p) for p in jpg_paths]
    Z = sample_Z(jpg_images, stride=STRIDE)
    del jpg_images  

    g_per_scheme = {}
    for name, wf in WEIGHT_SCHEMES.items():
        is_photon = name == "photon"
        g, _ = recover_g(Z, t, LAMBDA, wf, is_photon=is_photon)
        g_per_scheme[name] = g
        print(f"  g[{name}] done")

    # (set_name, image_paths, g_lookup) 
    sets = [
        ("raw", tiff_paths, None),
        ("jpg", jpg_paths, g_per_scheme),
    ]
    modes = ["linear", "log"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(sets) * len(modes) * len(WEIGHT_SCHEMES)
    done = 0
    t0 = time.time()

    for set_name, paths, g_lookup in sets:
        for mode in modes:
            for scheme_name, wf in WEIGHT_SCHEMES.items():
                is_photon = scheme_name == "photon"
                g = None if g_lookup is None else g_lookup[scheme_name]

                start = time.time()
                hdr = merge_stack(paths, t, wf, is_photon=is_photon, g=g, mode=mode)
                elapsed = time.time() - start

                out_name = f"door_{set_name}_{mode}_{scheme_name}.hdr"
                out_path = OUT_DIR / out_name
                writeHDR(str(out_path), hdr)

                done += 1
                print(f"[{done}/{total}] {out_name}  ({elapsed:.1f}s)  "
                      f"min={hdr.min():.4g} max={hdr.max():.4g} mean={hdr.mean():.4g}")

    print(f"\nall {total} HDR images written to {OUT_DIR}  (total {time.time()-t0:.1f}s)")
