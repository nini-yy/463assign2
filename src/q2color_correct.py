from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hw2 import (
    patch_avg_rgb,
    colorchecker_groundtruth_ordered,
    solve_color_affine,
    apply_color_affine,
    wb_from_patch,
    gamma_encode,
)
from cp_hw2 import readHDR, writeHDR
from display_fn import auto_scale, save_preview

HDR_DIR = Path(__file__).resolve().parent / "deliverables" / "q1" / "hdr_all16"
DELIVERABLES_DIR = Path(__file__).resolve().parent / "deliverables" / "q2"
COORDS_PATH = DELIVERABLES_DIR / "color_checker_coords.npy"

SELECTED_HDR = HDR_DIR / "door_jpg_log_uniform.hdr"

# ginput get patch + check pack manually
def get_patch_centers(hdr):
    if COORDS_PATH.exists():
        print(f"loading saved patch coordinates from {COORDS_PATH}")
        return np.load(COORDS_PATH)
    
    preview = np.clip(gamma_encode(hdr * auto_scale(hdr)), 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(preview)
    ax.set_title(
        "Click the 24 color checker patches in raster order\n"
        "(left->right, top->bottom). Last click = white patch, bottom-right."
    )
    centers = plt.ginput(24, timeout=0)
    plt.close(fig)

    centers = np.array(centers)
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(COORDS_PATH, centers)
    print(f"saved patch coords to {COORDS_PATH}")
    return centers


if __name__ == "__main__":
    hdr = readHDR(str(SELECTED_HDR))

    centers = get_patch_centers(hdr)

    # step 1: average RGB per patch
    measured = patch_avg_rgb(hdr, centers)

    # steps 2-3: homogeneous coords + affine least-squares fit
    ground_truth = colorchecker_groundtruth_ordered()
    M = solve_color_affine(measured, ground_truth)

    # step 4: apply affine transform, clip negs
    corrected = apply_color_affine(hdr, M)

    # step 5: wb using the white patch
    white_center = centers[-1]
    balanced = wb_from_patch(corrected, white_center)

    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DELIVERABLES_DIR / "door_color_corrected.hdr"
    writeHDR(str(out_path), balanced)
    print(f"wrote {out_path}")

    before_path = DELIVERABLES_DIR / "door_before_correction_preview.png"
    after_path = DELIVERABLES_DIR / "door_after_correction_preview.png"
    save_preview(hdr, before_path)
    save_preview(balanced, after_path)
    print(f"wrote {before_path}\nwrote {after_path}")
