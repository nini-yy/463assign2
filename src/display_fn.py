import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from hw2 import gamma_encode


def to_display(tm):
    return np.clip(gamma_encode(np.clip(tm, 0, 1)), 0, 1)


def auto_scale(hdr, percentile=99.0):
    p = np.percentile(hdr, percentile)
    return 0.9 / p if p > 0 else 1.0


def save_preview(image, path):
    scale = auto_scale(image)
    ldr = np.clip(gamma_encode(image * scale), 0, 1)
    imsave(path, (ldr * 255).astype(np.uint8))


def save_tonemap(tm, path):
    """gamma-encodes a tonemapped image, saves it, and returns the display-ready image"""
    img = to_display(tm)
    plt.imsave(path, img)
    print(f"wrote {path}")
    return img


def make_grid(row_fns, row_labels, col_values, col_label_fmt, out_path, figsize=(12, 8), dpi=150):
    """rows: one fn per row, called with each col value -> tonemapped image"""
    fig, axes = plt.subplots(len(row_fns), len(col_values), figsize=figsize)
    for r, (fn, row_label) in enumerate(zip(row_fns, row_labels)):
        for c, val in enumerate(col_values):
            ax = axes[r, c]
            tm = fn(val)
            ax.imshow(to_display(tm))
            ax.set_title(col_label_fmt(val), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(row_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pixel_histograms(samples, pixels, out_path, channel=1, bins=15):
    """samples: (num_frames, num_pixels, 3) values of each pixel across the ramp frames
       one histogram per pixel, with a gaussian of the same mean/std drawn on top"""
    names = "RGB"
    fig, axes = plt.subplots(1, len(pixels), figsize=(4.5 * len(pixels), 3.8))
    for ax, p, (y, x) in zip(axes, range(len(pixels)), pixels):
        vals = samples[:, p, channel]
        mu, sd = vals.mean(), vals.std(ddof=1)
        counts, edges, _ = ax.hist(vals, bins=bins, color="tab:gray", alpha=0.7)

        # gaussian with the same mean/std, scaled to histogram counts
        xs = np.linspace(edges[0], edges[-1], 200)
        bin_w = edges[1] - edges[0]
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(xs, pdf * len(vals) * bin_w, color="tab:red", label="gaussian fit")

        ax.set_title(f"pixel (y={y}, x={x}), {names[channel]}\n"
                     f"mean={mu:.0f}  std={sd:.1f}  var/mean={sd**2 / mu:.2f}", fontsize=10)
        ax.set_xlabel("value (16-bit, dark-frame subtracted)")
        ax.set_ylabel("# frames")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def single_vs_hdr(jpg_paths, picks, final_tm, out_path, stride=4):
    """shows the final tonemap is better than any single exposure
       picks: indices into jpg_paths to show next to the HDR"""
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(5 * (len(picks) + 1), 4))
    for ax, k in zip(axes, picks):
        ax.imshow(imread(jpg_paths[k])[::stride, ::stride])
        ax.set_title(f"single JPG: {jpg_paths[k].stem}")
    axes[-1].imshow(final_tm[::stride, ::stride])
    axes[-1].set_title("HDR, tonemapped")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
