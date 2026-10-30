"""Clean raw TIF/ND2 fluorescence images before analysis.

Removes diffuse background and autofluorescence, boosts local contrast,
denoises lightly and writes a standard RGB image (red = amastigotes,
blue = nuclei), so the analyzer always looks for parasites in red.

Usage:
  python image_cleaner.py                 # current folder
  python image_cleaner.py path/to/images  # a folder
  python image_cleaner.py image.tif       # one image

Cleaned images go to a "cleaned/" subfolder with a "_clean" suffix.
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile is missing. Install it with: pip install tifffile")
    sys.exit(1)

from skimage import filters, exposure


# settings
CONFIG = {
    # Background subtraction
    "bg_sigma_nuclei": 30,      # gaussian sigma for the nuclear background (px)
    "bg_sigma_amas": 8,         # sigma for the amastigote background (px)

    # normalisation
    "norm_low_pct": 40,         # low percentile (clips the background)
    "norm_high_pct": 99.8,      # high percentile (keeps the peaks)

    # local contrast enhancement (CLAHE)
    "use_clahe": True,
    "clahe_clip": 0.02,

    # denoising
    "denoise_sigma": 0.6,       # light gaussian blur to remove noise

    # channels for RGB / multichannel images
    # None = auto-detect, or give the indices
    "nuclei_channel": None,     # nuclear channel (blue / DAPI)
    "amas_channel": None,       # amastigote channel
}


def _load_image(filepath):
    """Load a TIF or ND2 file; returns (array, channel names)."""
    ext = Path(filepath).suffix.lower()
    channel_names = None

    if ext == ".nd2":
        try:
            import nd2
            with nd2.ND2File(filepath) as f:
                img = f.asarray()
                try:
                    channel_names = [c.channel.name for c in f.metadata.channels]
                except Exception:
                    channel_names = None
        except ImportError:
            print("  ERROR: nd2 is missing. Install it with: pip install nd2")
            return None, None
    elif ext in (".tif", ".tiff"):
        img = tifffile.imread(filepath)
    else:
        return None, None

    return img, channel_names


def _get_channels(img, channel_names):
    """Split the image into nuclear and amastigote channels.

    Returns (nuclei_raw, amas_raw, layout_info).
    """
    arr = img.astype(np.float32)
    if arr.ndim == 4:
        arr = arr.max(axis=0)

    # work out the channel layout
    if arr.ndim == 2:
        # single-channel image: use it for both (rare)
        return arr, arr, "single"

    # multichannel: (C,H,W) or (H,W,C)
    if arr.shape[0] <= 8:        # (C, H, W)
        n_ch = arr.shape[0]
        get = lambda i: arr[i]
        channels_last = False
    else:                        # (H, W, C)
        n_ch = arr.shape[2]
        get = lambda i: arr[:, :, i]
        channels_last = True

    # drop the alpha channel of RGBA images: a typical alpha is saturated
    # and featureless, and it would win the "brightest channel" heuristic
    # and steal the nuclei
    if n_ch == 4:
        alpha = get(3)
        if alpha.min() >= 250 or alpha.std() < 1.0:
            n_ch = 3

    # already an RGB composite (colour TIF)
    # Our own convention when saving: R = amastigotes, B = nuclei.
    # For a plain 3-channel RGB image without ND2 names, use it directly
    # instead of guessing from intensities.
    if n_ch == 3 and channels_last and not channel_names:
        r = get(0); g = get(1); b = get(2)
        # nuclei = blue; amastigotes = red or green, whichever has real signal
        #
        # Do NOT decide from peak/mean: an almost empty channel (e.g. green with
        # only noise) has a mean near 0 and therefore a huge peak/mean ratio.
        # That used to pick the empty channel and wipe out every amastigote
        # (images C13/A15).
        #
        # Instead, count strong signal: the number of pixels above a clear
        # threshold. A channel with real parasites has tens of thousands of
        # bright pixels; an empty one has almost none. Two thresholds make it
        # more stable.
        def _signal_strength(ch):
            # pixels clearly above the usual background noise
            return int((ch > 50).sum()) + int((ch > 80).sum())

        sr = _signal_strength(r)
        sg = _signal_strength(g)
        if max(sr, sg) < 20:
            # no channel has real amastigote signal: take the brightest
            amas, amas_name = (r, "ch0(R)") if r.mean() >= g.mean() else (g, "ch1(G)")
        elif sr >= sg:
            amas, amas_name = r, "ch0(R)"
        else:
            amas, amas_name = g, "ch1(G)"
        return b, amas, f"RGB composite: nuc=ch2(B), amas={amas_name}"

    # ND2: detect channels from their names
    nuc_idx, amas_idx = None, None
    if channel_names:
        for i, name in enumerate(channel_names):
            nl = name.lower()
            if any(x in nl for x in ['blue', 'dapi', 'hoechst', 'qblue', 'cy5']):
                nuc_idx = i
            elif any(x in nl for x in ['red', 'qred', 'green', 'gfp', 'fitc',
                                        'qgreen', 'cy3', 'tritc', 'alexa']):
                amas_idx = i

    # no names: fall back to intensities
    if nuc_idx is None or amas_idx is None:
        means = [float(get(i).mean()) for i in range(n_ch)]
        maxs  = [float(get(i).max()) for i in range(n_ch)]

        # nuclei = brightest on average (DAPI is bright and widespread)
        if nuc_idx is None:
            nuc_idx = int(np.argmax(means))

        # amastigotes = strongest point-like signal (high max/mean), skipping
        # nearly empty channels (max < 40 is just noise)
        if amas_idx is None:
            candidates = [i for i in range(n_ch)
                          if i != nuc_idx and maxs[i] >= 40]
            if candidates:
                # among channels with real signal, the highest relative peak
                ratios = [(maxs[i] / (means[i] + 1e-6), i) for i in candidates]
                amas_idx = max(ratios)[1]
            else:
                # nothing clear: take the second brightest
                order = sorted(range(n_ch), key=lambda i: means[i], reverse=True)
                amas_idx = order[1] if len(order) > 1 else nuc_idx

    return get(nuc_idx), get(amas_idx), f"nuc=ch{nuc_idx}, amas=ch{amas_idx}"


def _clean_nuclei(ch):
    """Nuclear channel: remove diffuse background, enhance, normalise."""
    ch = ch.astype(np.float32)

    # 1. background subtraction (large gaussian)
    bg = filters.gaussian(ch, sigma=CONFIG["bg_sigma_nuclei"], preserve_range=True)
    corr = np.clip(ch - bg, 0, None)

    # 2. light denoising
    if CONFIG["denoise_sigma"] > 0:
        corr = filters.gaussian(corr, sigma=CONFIG["denoise_sigma"], preserve_range=True)

    # 3. percentile normalisation
    if corr.max() > 0:
        lo = float(np.percentile(corr, CONFIG["norm_low_pct"]))
        hi = float(np.percentile(corr, CONFIG["norm_high_pct"]))
        norm = np.clip((corr - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        norm = corr

    # 4. CLAHE for local contrast
    if CONFIG["use_clahe"] and norm.max() > 0:
        norm = exposure.equalize_adapthist(norm, clip_limit=CONFIG["clahe_clip"])

    return (norm * 255).astype(np.uint8)


def _clean_amastigotes(ch):
    """Amastigote channel: enhance bright spots, remove background."""
    ch = ch.astype(np.float32)

    # 1. strong background subtraction (small sigma for spots)
    bg = filters.gaussian(ch, sigma=CONFIG["bg_sigma_amas"], preserve_range=True)
    corr = np.clip(ch - bg, 0, None)

    # 2. noise floor: drop diffuse background, keep real peaks only
    cvals = corr[corr > 0]
    if len(cvals) > 10:
        p99 = float(np.percentile(corr, 99))
        noise_floor = max(15.0, p99 * 0.5)
        # normalise keeping only the signal above the noise floor
        hi = max(float(np.percentile(corr, 99.9)), noise_floor + 10)
        norm = np.clip((corr - noise_floor) / (hi - noise_floor + 1e-6), 0, 1)
        norm_arr = (norm * 255)
        norm_arr[corr < noise_floor] = 0
        return norm_arr.astype(np.uint8)
    else:
        return np.zeros_like(ch, dtype=np.uint8)


def _detect_blue_amastigotes(blue_raw, red_raw, amas_clean):
    """Find amastigotes that stain BLUE and are missing from the red channel.

    Some parasites take up DAPI/Hoechst through their own DNA. Returns a
    uint8 mask with only the pure blue amastigotes, ready to be merged into
    the red channel.

    Deliberately conservative: spots must be compact, bright and about the
    size of an amastigote, so nuclear chromatin texture is not mistaken for
    parasites, and any blue spot that already has red signal on top is
    skipped so the same parasite is not counted twice.

    Filter settings:
      - top-hat with a disk of radius 7 (small objects up, large diffuse
        nuclei ignored)
      - 99th percentile threshold
      - area 5-120 px^2, compact shape (solidity > 0.80, eccentricity < 0.88)
      - mean blue intensity > 60
    """
    from skimage import measure, morphology
    from skimage.morphology import white_tophat, disk
    import numpy as _np

    blue = blue_raw.astype(_np.float32)
    red  = red_raw.astype(_np.float32)

    # top-hat: small bright spots over the diffuse nuclear background
    th = white_tophat(blue, disk(7))
    if th.max() <= 0:
        return _np.zeros_like(blue, dtype=_np.uint8)

    thr = max(float(_np.percentile(th, 99.0)), 16.0)
    pts = th > thr
    pts = morphology.remove_small_objects(pts, 4)
    if not pts.any():
        return _np.zeros_like(blue, dtype=_np.uint8)

    lbl = measure.label(pts)
    props = measure.regionprops(lbl, intensity_image=blue)

    # strong red signal = an amastigote already detected in red;
    # both the raw and the cleaned red channel are used to avoid duplicates
    red_strong = (red > 50) | (amas_clean > 0)

    out = _np.zeros_like(blue, dtype=_np.uint8)
    n_pure = 0
    for p in props:
        if not (5 <= p.area <= 120):
            continue
        if p.eccentricity >= 0.88 or p.solidity <= 0.80:
            continue
        if p.mean_intensity <= 60:
            continue
        # red signal on top? already counted in red, skip it
        cy, cx = [int(v) for v in p.centroid]
        y0, y1 = max(0, cy - 3), cy + 4
        x0, x1 = max(0, cx - 3), cx + 4
        if red_strong[y0:y1, x0:x1].any():
            continue
        # pure blue amastigote: write it at a fixed high intensity (like a
        # sharp red one) so the detector counts it the same way
        for yy, xx in p.coords:
            out[yy, xx] = 255
        n_pure += 1

    return out


def clean_image(filepath, output_dir):
    """Clean one image and save it in output_dir."""
    name = Path(filepath).stem
    print(f"\n-> {Path(filepath).name}")

    img, channel_names = _load_image(filepath)
    if img is None:
        print("  unsupported format")
        return False

    nuc_raw, amas_raw, layout = _get_channels(img, channel_names)
    print(f"  channels: {layout}")
    if channel_names:
        print(f"  ND2 names: {channel_names}")

    # clean each channel
    nuc_clean  = _clean_nuclei(nuc_raw)
    amas_clean = _clean_amastigotes(amas_raw)

    print(f"  nuclei cleaned: max={nuc_clean.max()}, std={nuc_clean.std():.1f}")
    print(f"  amastigotes cleaned: pixels>0={100*(amas_clean>0).sum()/amas_clean.size:.2f}%")

    # output RGB image:
    #   channel 0 (R) = amastigotes
    #   channel 2 (B) = nuclei
    h, w = nuc_clean.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = amas_clean   # red = amastigotes
    rgb[:, :, 2] = nuc_clean    # blue = nuclei

    # save
    out_path = Path(output_dir) / f"{name}_clean.tif"
    tifffile.imwrite(str(out_path), rgb)
    print(f"  saved {out_path.name}")
    return True


def clean_image_to_path(filepath, out_path, verbose=False):
    """Clean ONE image and write it exactly to out_path (used by the GUI).

    The output always follows the analyzer convention: R (0) = amastigotes
    (whether they were red or green originally) and B (2) = nuclei, so the
    analysis can always look for parasites in red.

    Returns a dict of metrics, or None if the format is not supported.
    """
    img, channel_names = _load_image(str(filepath))
    if img is None:
        return None

    nuc_raw, amas_raw, layout = _get_channels(img, channel_names)
    if verbose:
        print(f"  channels: {layout}")

    nuc_clean  = _clean_nuclei(nuc_raw)
    amas_clean = _clean_amastigotes(amas_raw)

    # Find pure BLUE amastigotes (missing from the red channel) and merge
    # them into the amastigote channel, so they count like red ones,
    # without duplicating those that already have red signal.
    blue_amas = _detect_blue_amastigotes(nuc_raw, amas_raw, amas_clean)
    n_blue = int((blue_amas > 0).sum())
    if n_blue > 0:
        amas_clean = np.maximum(amas_clean, blue_amas)
        # Remove them from the NUCLEAR channel too: left there, their blue
        # signal forms a swarm that CellPose would merge into one huge fake
        # nucleus. Once removed, CellPose no longer mistakes them for nuclear
        # material and they are counted only as amastigotes in red.
        nuc_clean = nuc_clean.copy()
        nuc_clean[blue_amas > 0] = 0
        if verbose:
            print(f"  blue amastigotes added: {n_blue} px (removed from the nuclear channel)")

    h, w = nuc_clean.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = amas_clean   # red = amastigotes (standardised)
    rgb[:, :, 2] = nuc_clean    # blue = nuclei

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), rgb)

    amas_pct = 100.0 * (amas_clean > 0).sum() / amas_clean.size
    return {
        "out_path": str(out_path),
        "layout": layout,
        "nuclei_std": float(nuc_clean.std()),
        "amas_pixels_pct": float(amas_pct),
        "blue_amas_added": n_blue,
    }


def main():
    # input: a file or a folder
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "."

    target_path = Path(target)

    # collect the files to process
    if target_path.is_file():
        files = [target_path]
        base_dir = target_path.parent
    elif target_path.is_dir():
        files = sorted([f for f in target_path.iterdir()
                        if f.suffix.lower() in (".tif", ".tiff", ".nd2")])
        base_dir = target_path
    else:
        print(f"ERROR: '{target}' does not exist")
        return

    if not files:
        print(f"No TIF/ND2 images found in {target}")
        return

    # output folder
    output_dir = base_dir / "cleaned"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"Image cleaner: {len(files)} file(s)")
    print(f"Output: {output_dir}")
    print("=" * 70)

    ok = 0
    for f in files:
        try:
            if clean_image(f, output_dir):
                ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print(f"Done: {ok}/{len(files)} images cleaned")
    print(f"Saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
