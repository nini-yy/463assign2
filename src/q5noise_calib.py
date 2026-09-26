from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import hw2
from hw2 import (RAMP_BOX, STRIDE, OUT_DIR, calib_paths,
                 compute_dark_frame, load_ramp_frames,
                 pixel_mean_var, mean_variance_curve, fit_noise_line,
                 merge_stack, w_optimal, photographic_tonemap_luminance, gamma_encode)
from cp_hw2 import readHDR, writeHDR


# --------------------------------------------------------
# NOISE CALIBRATION

# pix for histogram
HIST_PIXELS = [(2000, 1200), (2000, 2200), (2000, 3300), (2000, 5100)]
MIN_COUNT = 20 # a mean value needs this many pixels for its average variance to count

#  reduce the range of mean values you use for your fit to just the lower mean values
FIT_MAX = {"R": 2600, "G": 1500, "B": 700}

# part 4 RAW stack:
MY_STACK = Path(__file__).resolve().parent.parent / "data" / "my_stack"
NUM_EXPOSURES = 10
RAW_ZMIN = 0.005 # same as part 4 RAW 
Q4_DIR = Path(__file__).resolve().parent / "deliverables" / "q4"

# part 4 picks
K, B = 0.15, 0.95

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. dark frame = average of the lens-cap frames
dark_path = OUT_DIR / "dark_frame.npy"
if dark_path.exists():
    dark = np.load(dark_path)
else:
    dark = compute_dark_frame(calib_paths("dark"))
    np.save(dark_path, dark)

# 2. dark-subtracted ramp frames
frames = load_ramp_frames(dark) # (num_frames, h, w, 3)

# 3. one histogram per pixel (green channel) across the ramp frames
y0, y1, x0, x1 = RAMP_BOX
fig, axes = plt.subplots(1, len(HIST_PIXELS), figsize=(16, 3.5))
for (y, x), ax in zip(HIST_PIXELS, axes):
    vals = frames[:, (y - y0) // STRIDE, (x - x0) // STRIDE, 1] # full-res -> cropped/strided index
    ax.hist(vals, bins=12)
    ax.set_title(f"pixel {(y, x)}\nmean {vals.mean():.0f}, std {vals.std(ddof=1):.1f}")
    ax.set_xlabel("value (DN)")
plt.tight_layout()
plt.savefig(OUT_DIR / "pixel_histograms.png")
plt.show()

# 4. per-pixel mean
mu, var = pixel_mean_var(frames)
del frames

# 5. per channel: round means, average variance per mean, fit
noise = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for c, name in enumerate(["R", "G", "B"]):
    means, avg_var = mean_variance_curve(mu[..., c], var[..., c], MIN_COUNT)
    g, s2_add = fit_noise_line(means, avg_var, FIT_MAX[name])
    noise[name] = (g, s2_add)
    print(f"{name}: g = {g:.4f},  sigma^2_additive = {s2_add:.2f}")

    fit = means <= FIT_MAX[name]
    ax = axes[c]
    ax.scatter(means, avg_var, s=1, alpha=0.3, color=name.lower())
    ax.plot(means, g * means + s2_add, "k--", lw=0.8)   # fitted line extended over all means
    ax.plot(means[fit], g * means[fit] + s2_add, "k")   # the range it was fit on
    ax.set_title(f"{name}: g = {g:.3f}, $\\sigma^2_{{add}}$ = {s2_add:.1f}")
    ax.set_xlabel("mean (DN)")
    ax.set_ylabel("variance (DN$^2$)")

plt.tight_layout()
plt.savefig(OUT_DIR / "mean_variance.png")
plt.show()

# save noise for merging 
np.save(OUT_DIR / "noise_params.npy", noise)


# --------------------------------------------------------
# MERGE FOR OPTIMAL

# 6. merge the part 4 RAW stack: dark-frame subtraction + noise-optimal weights (Eq. 15)
# g, sigma^2 fitted in 16-bit DN -> [0, 1] units to match z; reported values, not clamped
hw2.ZMIN = RAW_ZMIN # weight fn + zero-weight repair read this global
g = np.array([noise[c][0] for c in "RGB"]) / 65535
s2_add = np.array([noise[c][1] for c in "RGB"]) / 65535 ** 2
t = np.array([2.0 ** (k - NUM_EXPOSURES) for k in range(1, NUM_EXPOSURES + 1)])
tiff_paths = [MY_STACK / f"exposure{k}.tiff" for k in range(1, NUM_EXPOSURES + 1)]

hdr_opt = merge_stack(tiff_paths, t, lambda z, t_k: w_optimal(z, t_k, g, s2_add),
                      is_photon=True, mode="linear", dark=dark)
writeHDR(str(OUT_DIR / "my_hdr_optimal.hdr"), hdr_opt)
del dark

# 7. tonemap and compare with my part 4 best (JPG)
def tonemap(hdr):
    return np.clip(gamma_encode(np.clip(photographic_tonemap_luminance(hdr, K=K, B=B), 0, 1)), 0, 1)

results = {
    "part 4 best (JPG, log, uniform)": tonemap(readHDR(str(Q4_DIR / "my_hdr_jpg.hdr"))),
    "part 5 RAW (dark-subtracted, optimal)": tonemap(hdr_opt),
}
del hdr_opt
plt.imsave(OUT_DIR / f"final_optimal_luminance_K{K}_B{B}.png", results["part 5 RAW (dark-subtracted, optimal)"])

fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.4 * (1)))
for col, (label, img) in enumerate(results.items()):
    axes[0, col].imshow(img[::4, ::4])
    axes[0, col].set_title(label, fontsize=11)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUT_DIR / "optimal_vs_part4.png", dpi=120)
plt.show()
