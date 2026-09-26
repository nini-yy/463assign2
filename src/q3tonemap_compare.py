from pathlib import Path

from hw2 import photographic_tonemap_rgb, photographic_tonemap_luminance
from cp_hw2 import readHDR
from display_fn import make_grid

DELIVERABLES_DIR = Path(__file__).resolve().parent / "deliverables" / "q3"

SELECTED_HDR = Path(__file__).resolve().parent / "deliverables" / "q2" / "door_color_corrected.hdr"

K_VALUES = [0.09, 0.15, 0.36]  # low/default/high key
B_FIXED = 0.95

B_VALUES = [0.8, 0.95, 1.0]  # low/default/no burn
K_FIXED = 0.15


if __name__ == "__main__":
    hdr = readHDR(str(SELECTED_HDR))

    # sweep K (key) at fixed B (burn)
    make_grid(
        row_fns=[
            lambda K: photographic_tonemap_rgb(hdr, K=K, B=B_FIXED),
            lambda K: photographic_tonemap_luminance(hdr, K=K, B=B_FIXED),
        ],
        row_labels=["RGB", "Luminance"],
        col_values=K_VALUES,
        col_label_fmt=lambda K: f"K={K}, B={B_FIXED}",
        out_path=DELIVERABLES_DIR / "tonemap_K_sweep.png",
    )

    # sweep B (burn) at fixed K (key)
    make_grid(
        row_fns=[
            lambda B: photographic_tonemap_rgb(hdr, K=K_FIXED, B=B),
            lambda B: photographic_tonemap_luminance(hdr, K=K_FIXED, B=B),
        ],
        row_labels=["RGB", "Luminance"],
        col_values=B_VALUES,
        col_label_fmt=lambda B: f"K={K_FIXED}, B={B}",
        out_path=DELIVERABLES_DIR / "tonemap_B_sweep.png",
    )
