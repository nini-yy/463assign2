import numpy as np
import subprocess

from pathlib import Path
from skimage.io import imread

from cp_hw2 import read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "door_stack"
NUM_EXPOSURES = 16
ZMIN, ZMAX = 0.05, 0.95 # clipping

CALIB_DIR = Path(__file__).resolve().parent.parent / "data" / "noise_calib"
OUT_DIR = Path(__file__).resolve().parent / "deliverables" / "q5"

# ----------------------------------------------------
# "weighting schemes"

# if Zmin <= z < Zmax: 1.0 
# else: 0
def w_uniform(z):
    z = np.asarray(z, dtype=float)
    return np.where((z >= ZMIN) & (z <= ZMAX), 1.0, 0.0)

# if Zmin <= z < Zmax: min(z, 1-z) 
# else: 0
def w_tent(z):
    z = np.asarray(z, dtype=float)
    return np.where((z >= ZMIN) & (z <= ZMAX), np.minimum(z, 1.0 - z), 0.0)

# if Zmin <= z < Zmax: exp(-4 (z - 0.5)^2 / 0.5^2)
# else: 0
def w_gaussian(z):
    z = np.asarray(z, dtype=float)
    return np.where((z >= ZMIN) & (z <= ZMAX), np.exp(-4.0 * (z - 0.5) ** 2 / 0.5 ** 2), 0.0)

# if Zmin <= z < Zmax: t^k
# else: 0
def w_photon(z, t_k):
    z = np.asarray(z, dtype=float)
    weight = np.full_like(z, float(t_k)) # builds size z vec with t_k vals
    return np.where((z >= ZMIN) & (z <= ZMAX), weight, 0.0)

WEIGHT_SCHEMES = {
    "uniform": w_uniform,
    "tent": w_tent,
    "gaussian": w_gaussian,
    "photon": w_photon,
}

# ----------------------------------------------------
# strided downsampling: pick every n = 200 

def sample_Z(images, stride=200):
    """input: image as list of (H, W, 3) uint8 arrays, one per exposure
       returns: Z of shape (num_samples, num_images)."""
    sub = [img[::stride, ::stride, :] for img in images]
    stacked = np.stack(sub, axis=0)  # (num_images, h', w', 3)
    N = stacked.shape[0]
    Z = stacked.reshape(N, -1).T  # (h'*w'*3, num_images)
    return Z


# ----------------------------------------------------
# LINEARIZE RENDERED IMAGE
# recovering g
#   rows:    [ data term: 256 g curve + N light slots]
#            [ anchor                                ]
#            [ regularization                        ]


def recover_g(Z, tk, lmda, weight_fn, is_photon=False):
    n = 256
    N, P = Z.shape

    num_data_rows = N * P
    num_reg_rows = n - 2  # z = 1 to 254; boundaries z=0, z=255 have no ∇2g
    num_rows = num_data_rows + 1 + num_reg_rows
    num_cols = n + N # g + N (one light val per sampled spot)

    A = np.zeros((num_rows, num_cols))
    b = np.zeros(num_rows)

    # data rows
    row = 0
    for i in range(N): # samples
        for k in range(P): # images
            z = int(Z[i, k])
            zn = z / 255.0 # I / 255
            wij = weight_fn(zn, tk[k]) if is_photon else weight_fn(zn)
            A[row, z] = wij # w*g(z)
            A[row, n + i] = -wij # for w*l(z)
            b[row] = wij * np.log(tk)[k] # for w*log(tk)
            row += 1

    # anchor row so not rank deficient (Debevec's)
    A[row, 128] = 1.0
    row += 1

    # regularization term
    # ∇2g(z) = λw g(z+ 1)−2λw g(z) + λw g(z−1)
    for z in range(1, n - 1):
        wz = 1.0 if is_photon else weight_fn(z / 255.0)
        A[row, z - 1] = lmda * wz # g(z−1)
        A[row, z] = -2.0 * lmda * wz # g(z)
        A[row, z + 1] = lmda * wz # g(z+ 1)
        row += 1

    # min ∥Av−b∥2
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    g = v[:n]
    le = v[n:]
    return g, le

#I = exp(g(I))
def linearize(images, g):
    return [np.exp(g[img]) for img in images]


# ----------------------------------------------------
# MERGE EXPOSURE STACK

def merge_stack(image_paths, tk, weight_fn, is_photon=False, g=None, mode="linear", eps=1e-8, stride=1,
                dark=None):
    """g: switch where true = RAW, false = JPG
       stride: >1 downsamples each image with [::stride, ::stride] (for fast debugging)
       dark: full-res dark frame (16-bit units); if given, subtracts (t_k / t_nc) * dark from each image
       is_photon: True for any weight_fn that also takes t_k (photon, optimal)"""
    k_dark = int(np.argmin(tk)) # shortest exposure (always over)
    k_bright = int(np.argmax(tk)) # longest exposure (always under)

    num = None
    den = None
    dark_ldr = None
    bright_ldr = None

    for k, path in enumerate(image_paths): #∑_k
        raw = imread(path)[::stride, ::stride]
        norm = 255.0 if raw.dtype == np.uint8 else 65535.0 # 2^16-1
        ldr = raw.astype(np.float64) / norm  # I^k_LDR to [0,1] norm
        if dark is not None:
            ldr = dark_subtract(ldr, dark[::stride, ::stride] / norm, tk[k])  # scaled by t_k / t_nc
        lin = ldr if g is None else np.exp(g[raw])  # I^k_lin: if RAW alr lin, ow apply exp(g(z))

        w = weight_fn(ldr, tk[k]) if is_photon else weight_fn(ldr)

        # init to img shape on the first itr
        if num is None:
            num = np.zeros(ldr.shape)
            den = np.zeros(ldr.shape)

        # which type of merging 
        if mode == "linear":
            num += w * (lin / tk[k])
        elif mode == "log":
            num += w * (np.log(lin + eps) - np.log(tk)[k])
        else:
            raise ValueError(f"unknown mode {mode!r}")
        den += w

        # check darkest & lightest
        if k == k_dark:
            dark_ldr = ldr
        if k == k_bright:
            bright_ldr = ldr

    safe_den = np.where(den == 0, 1.0, den) # check for every OB pix (weight 0 cannot do div)
    hdr = (num / safe_den) if mode == "linear" else np.exp(num / safe_den)

    # repairs for zero weights
    overexposed = (den == 0) & (dark_ldr > ZMAX)
    underexposed = (den == 0) & (bright_ldr < ZMIN)
    for c in range(hdr.shape[-1]):
        valid_c = hdr[..., c][~(den == 0)[..., c]]
        hi = valid_c.max() if valid_c.size else 1.0 # find highest val to fill with
        lo = valid_c.min() if valid_c.size else 0.0 # lowest
        hdr[..., c] = np.where(overexposed[..., c], hi, hdr[..., c]) # fill
        hdr[..., c] = np.where(underexposed[..., c], lo, hdr[..., c])

    return hdr


# ----------------------------------------------------
# COLOR CORRECTION

# step 1: avg RGB from each patch
def patch_avg_rgb(image, centers, half_size=15):
    """centers: ginput array of picks -- 24
       returns RGB per patch -- (24, 3)"""
    
    avgs = np.zeros((len(centers), 3))

    for i, (x, y) in enumerate(centers):
        x, y = int(round(x)), int(round(y))
        crop = image[y - half_size:y + half_size, x - half_size:x + half_size]
        avgs[i] = crop.reshape(-1, 3).mean(axis=0)

    return avgs

# HELPER (make sure that we pick the right square)
def colorchecker_groundtruth_ordered():
    r, g, b = read_colorchecker_gm() # (4, 6)
    gt = np.stack([r, g, b], axis=-1) # (4, 6, 3)
    gt = gt[:, ::-1, :].transpose(1, 0, 2) # (6, 4, 3), white -> bottom-right
    gt = gt.reshape(-1, 3) # (24, 3)

    white_idx = int(np.argmax(gt.sum(axis=1)))
    if white_idx != len(gt) - 1:
        print(f"WARNING: expected the white patch last. CHECK ORDER OF CLICKS")
    return gt


# steps 2-3: homogeneous coords + affine least-squares fit
def solve_color_affine(measured_rgb, ground_truth_rgb):
    ones = np.ones((measured_rgb.shape[0], 1))
    A = np.hstack([measured_rgb, ones])  # (24, 4) homogeneous coords

    M, *_ = np.linalg.lstsq(A, ground_truth_rgb, rcond=None) # min ||AM - B|}^2
    return M


# step 4: apply affine transform, clip negatives
def apply_color_affine(image, M):
    H, W, C = image.shape

    flat = image.reshape(-1, C) # (H W, 3)
    ones = np.ones((flat.shape[0], 1)) 
    flat_hom = np.hstack([flat, ones]) # (H W, 4)

    out = flat_hom @ M # (H w, 3)

    out = np.clip(out, 0, None) # clip negatives to 0
    return out.reshape(H, W, C) # (H, W, 3)


# step 5: wb so the white patch RGB equal
def wb_from_patch(image, white_center, half_size=15):
    # repeat from step 1
    x, y = int(round(white_center[0])), int(round(white_center[1]))
    crop = image[y - half_size:y + half_size, x - half_size:x + half_size]
    white_avg = crop.reshape(-1, 3).mean(axis=0)

    scale = white_avg.mean() / white_avg # mean() = 1/3 (rw, gw, bw); scale = (m/rw, m/gw, m/bw)
    return image * scale # (m, m, m) neutral white


# ----------------------------------------------------
# PHOTOGRAPHIC TONEMAPPING

# equation 11: log mean I_m,HDR
def log_avg_luminance(x, eps=1e-8): # Reinhardt
    return np.exp(np.mean(np.log(x + eps)))

# global tone mapping operator for medium dynamic range images: I_ijTM
# apply it by tonemapping all color channels simultaneously in the same way
def photographic_tonemap_rgb(hdr, K=0.15, B=0.95, eps=1e-8):
    I_mHDR = log_avg_luminance(hdr, eps)

    # each pixel is scaled such that the log average luminance is mapped to the key value 
    I_ijHDR = (K / I_mHDR) * hdr

    # smallest luminance that is mapped to pure white
    I_white = B * I_ijHDR.max()

    return I_ijHDR * (1.0 + I_ijHDR / I_white ** 2) / (1.0 + I_ijHDR)

# apply it only to the luminance channel Y
def photographic_tonemap_luminance(hdr, K=0.15, B=0.95, eps=1e-8):
    # tonemap brightness only
    XYZ = lRGB2XYZ(hdr) # convert to XYZ (Y is lunimance)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    denom = np.maximum(X + Y + Z, eps)
    x, y = X / denom, Y / denom # standard xyY chromaticity coordinates

    I_mHDR = log_avg_luminance(Y, eps)
    I_ijHDR = (K / I_mHDR) * Y
    Y_white = B * I_ijHDR.max()
    Y_ijTM = I_ijHDR * (1.0 + I_ijHDR / Y_white ** 2) / (1.0 + I_ijHDR)

    ratio = Y_ijTM / np.maximum(y, eps)  # = X+Y+Z after tonemapping, since x,y fixed
    X_ijTM = ratio * x
    Z_ijTM = ratio * (1.0 - x - y)

    return XYZ2lRGB(np.stack([X_ijTM, Y_ijTM, Z_ijTM], axis=-1)) # invert the color transform back to RGB


def gamma_encode(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)

# ----------------------------------------------------
# NOISE CALIBRATION
ISO = 100 # from part 4
SHUTTER = {100: 1/2, 200: 1/4, 400: 1/8, 800: 1/16, 1600: 1/32}
RAMP_SKIP = 5 # skip ramp frames bc light went out so darker
STRIDE = 5 # every how many pix to skip

# (y0, y1, x0, x1): the ramp paper in the frame
RAMP_BOX = (400, 3600, 1000, 5200)


# HELPERS:
def calib_paths(kind, iso=ISO):
    """kind: "dark" or "ramp" -> sorted NEFs in CALIB_DIR/kind/iso{iso}"""
    return sorted((CALIB_DIR / kind / f"iso{iso}").glob("*.nef"))

def crop(image, box=RAMP_BOX, stride=STRIDE):
    y0, y1, x0, x1 = box
    return image[y0:y1:stride, x0:x1:stride]

# run dcraw so we dont have to store all the data.
def develop_raw(nef_path):
    out = subprocess.run(["dcraw", "-c", "-w", "-o", "1", "-q", "3", "-4", str(nef_path)],
                         capture_output=True).stdout
    _, size, _, data = out.split(b"\n", 3) 
    w, h = map(int, size.split())
    return np.frombuffer(data, dtype=">u2").reshape(h, w, 3).astype(np.float32)


# MAIN:  

# dark frame = average the images
def compute_dark_frame(dark_paths):
    return sum(develop_raw(p) for p in dark_paths) / len(dark_paths)

# dark frame subtraction: subtract from each image the dark frame you computed
def dark_subtract_noise(image, dark):
    return image - dark


# stack of dark-subtracted ramp frames (crop and stride applied)
def load_ramp_frames(dark, box=RAMP_BOX, stride=STRIDE):
    """dark: full-res dark frame -> (num_frames, h, w, 3)"""
    dark = crop(dark, box, stride)
    frames = [dark_subtract(crop(develop_raw(p), box, stride), dark)
              for p in calib_paths("ramp")[RAMP_SKIP:]]
    return np.array(frames)


# per-pixel mean and variance
def pixel_mean_var(frames):
    return frames.mean(axis=0), frames.var(axis=0, ddof=1)


# round the mean to the nearest integer,
def mean_variance_curve(mu_c, var_c, min_count):
    m = np.round(mu_c).astype(int).ravel()
    v = var_c.ravel()
    ok = m >= 0
    m, v = m[ok], v[ok]

    counts = np.bincount(m)                   # pixels per mean value
    sums = np.bincount(m, weights=v)          # total variance per mean value
    means = np.nonzero(counts >= min_count)[0]
    return means, sums[means] / counts[means]


# fit variance = g * mean + sigma^2_additive
def fit_noise_line(means, avg_var, fit_max):
    """fits only means <= fit_max -> (g, sigma^2_additive)"""
    fit = means <= fit_max
    g, s2_add = np.polyfit(means[fit], avg_var[fit], 1) # slope = g, intercept = sigma^2_add
    return g, s2_add


# ----------------------------------------------------
# MERGING WITH OPTIMAL WEIGHTS

#
def dark_subtract(image, dark, t_k=None, t_nc=SHUTTER[ISO]):
    scale = t_k / t_nc
    return image - scale * dark

# if Zmin <= z <= Zmax: t_k^2 / (g z + sigma^2_add)
# else: 0
def w_optimal(z, t_k, g, s2_add):
    z = np.asarray(z, dtype=float)
    ok = (z >= ZMIN) & (z <= ZMAX)
    denom = np.where(ok, g * z + s2_add, 1.0)
    return np.where(ok, t_k ** 2 / denom, 0.0)


