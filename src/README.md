# HW2: HDR imaging, tonemapping, and noise calibration

## Setup

```bash
pip install -r ../requirements.txt   # numpy, scikit-image, matplotlib, opencv-python
brew install dcraw                   # needed for RAW development
```

Run every script from inside `src/`, e.g. `python3 q4my_photos.py`.

## Data layout

All data lives in `../data/`:

| Folder | Contents |
|---|---|
| `door_stack/` | provided stack: `exposure1..16.jpg`, `.nef`, `.tiff` |
| `my_stack/` | my Part 4 stack: `exposure1..10.jpg`, `.nef`, `.tiff` (1/512 s to 1 s, 1-stop steps, f/8, ISO 100, Daylight WB) |
| `noise_calib/dark/iso100/`, `noise_calib/ramp/iso100/` | Part 5 calibration: 50 lens-cap NEFs and 50 ramp NEFs, 0.5 s |

The `.tiff` files are linear 16-bit, developed from the NEFs with:

```bash
dcraw -w -o 1 -q 3 -4 -T exposure*.nef
```

(`-w` camera white balance, `-o 1` sRGB, `-q 3` AHD demosaicing, `-4` linear 16-bit, `-T` TIFF.)
The Part 5 calibration NEFs are developed on the fly with the same flags, so no TIFFs are stored for them.

## Files

| File | Purpose |
|---|---|
| `hw2.py` | Core functions for all parts: weighting schemes, g recovery (Debevec & Malik), HDR merging, color correction, photographic tonemapping, gamma encoding, and noise calibration (dark frame, dark subtraction, mean/variance, line fit, optimal weights) |
| `display_fn.py` | Helpers that save figures and preview images |
| `cp_hw2.py`, `cp_exr.py` | Provided course code (`readHDR`, `writeHDR`, color checker, RGB/XYZ conversion) |
| `q1p2linearize_g_plot.py` | Part 1: recover and plot g for each weighting scheme |
| `q1p3merge_to_HDR_16.py` | Part 1: build all 16 HDR variants (RAW/JPG × linear/log × 4 weights) |
| `q2color_correct.py` | Part 2: color-checker color correction and white balance |
| `q3tonemap_compare.py` | Part 3: photographic tonemapping sweeps (RGB and luminance) |
| `q4my_photos.py` | Part 4: HDR merge and tonemapping of my own stack |
| `q5noise_calib.py` | Part 5: noise calibration and merging with noise-optimal weights |

Each `qN…` script writes its outputs to `deliverables/qN/`.

## Run order

Later parts read earlier outputs, so run in this order:

1. `q1p2linearize_g_plot.py`, `q1p3merge_to_HDR_16.py`
2. `q2color_correct.py`: uses `door_jpg_log_uniform.hdr` from Part 1. The first run asks you to click the 24 color-checker patches (in order, white patch last) and saves the coordinates for later runs.
3. `q3tonemap_compare.py`: uses the color-corrected HDR from Part 2.
4. `q4my_photos.py`
5. `q5noise_calib.py`: uses the Part 4 JPG HDR for its comparison figure.

