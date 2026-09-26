import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread

from hw2 import DATA_DIR, NUM_EXPOSURES, WEIGHT_SCHEMES, sample_Z, recover_g

DELIVERABLES_DIR = Path(__file__).resolve().parent / "deliverables" / "q1"

if __name__ == "__main__":
    # exposure k has shutter speed (1/2048) * 2^(k-1), k = 1..16
    t = np.array([(1.0 / 2048.0) * 2.0 ** (k - 1) for k in range(1, NUM_EXPOSURES + 1)])

    images = [imread(DATA_DIR / f"exposure{k}.jpg") for k in range(1, NUM_EXPOSURES + 1)]

    # TODO: tune stride and lambda by eye against the g plot below
    stride = 200
    lam = 50.0

    Z = sample_Z(images, stride=stride)

    linestyles = {"uniform": "-", "tent": "--", "gaussian": ":", "photon": "-."}

    curves = {}
    for name, wf in WEIGHT_SCHEMES.items():
        is_photon = name == "photon"
        g, _ = recover_g(Z, t, lam, wf, is_photon=is_photon)
        curves[name] = g

    fig, ax = plt.subplots()
    for name, g in curves.items():
        ax.plot(g, label=name, linewidth=1.2, linestyle=linestyles[name])

    ax.set_xlabel("pixel value z")
    ax.set_ylabel("g(z) = log(exposure)")
    ax.legend(loc="lower right")
    ax.set_title(f"recovered response curve g (lambda={lam})")

    DELIVERABLES_DIR.mkdir(exist_ok=True)
    out_path = DELIVERABLES_DIR / "g_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
