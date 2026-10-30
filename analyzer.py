"""T. cruzi infection analysis: nucleus segmentation, amastigote detection
and assignment of parasites to host cells."""

import numpy as np
from pathlib import Path
import warnings


def _remove_small(binary, size):
    """remove_small_objects across skimage versions (max_size vs min_size)."""
    from skimage import morphology as sk_m
    try:
        return sk_m.remove_small_objects(binary, max_size=size)
    except TypeError:
        return sk_m.remove_small_objects(binary, min_size=size)
warnings.filterwarnings('ignore')

# --- hardware ---
def _check_gpu(force_mode="auto") -> bool:
    """Return True if CellPose should run on the GPU (force_mode: auto/gpu/cpu)."""
    if force_mode == "cpu":
        print("[Hardware] CPU forced by user")
        return False
    elif force_mode == "gpu":
        # Forcing GPU is not enough: the packaged build may ship PyTorch
        # without CUDA, and then CellPose fails. Check first and fall back
        # to CPU if the GPU is not usable.
        try:
            from cellpose import core
            if core.use_gpu():
                print("[Hardware] GPU forced by user")
                return True
        except Exception:
            pass
        print("[Hardware] GPU requested but not available (PyTorch without CUDA), "
              "using CPU")
        return False
    else:  # auto
        try:
            from cellpose import core
            gpu = core.use_gpu()
            if gpu:
                print("[Hardware] GPU detected")
            else:
                print("[Hardware] no GPU found, using CPU")
            return gpu
        except Exception:
            print("[Hardware] no GPU, using CPU")
            return False

USE_GPU = _check_gpu()

# --- CPU speed-up: segment nuclei at reduced resolution ---
# On CPU, CellPose time grows with pixel count. Nuclei are large (~60 px),
# so half resolution gives the same masks in about a quarter of the time.
# The mask is scaled back with nearest neighbour to keep integer labels.
# Only used on CPU, for large images whose nuclei are not tiny.
CPU_DOWNSCALE_ENABLED  = True
CPU_DOWNSCALE_MIN_DIM  = 1000     # only when the short side is larger than this
CPU_DOWNSCALE_TARGET   = 768      # target short side after downscaling
CPU_DOWNSCALE_MIN_DIAM = 24       # keep full resolution if nuclei are smaller than this

# --- fast CPU mode: classic watershed instead of CellPose ---
# cpsam (CellPose 4.x) takes minutes per image on CPU, which is not
# workable on lab computers without a GPU. A classic watershed does the
# same job in seconds for the round DAPI nuclei we usually see.
# On GPU, CellPose is always used.
CPU_FAST_NUCLEI = True   # True: watershed on CPU (fast)
                          # False: CellPose on CPU as well (slow)


def _compute_downscale_factor(shape, diameter, use_gpu):
    """Downscale factor for nucleus segmentation on CPU (1.0 = keep size).

    The target is aggressive (768 px) because cpsam is heavy on CPU; nuclei
    of ~50 px are still ~19 px after downscaling, which segments fine.
    """
    if use_gpu or not CPU_DOWNSCALE_ENABLED:
        return 1.0
    min_dim = min(shape[0], shape[1])
    if min_dim <= CPU_DOWNSCALE_MIN_DIM:
        return 1.0
    # small nuclei would disappear if we shrank them further
    if diameter is not None and diameter < CPU_DOWNSCALE_MIN_DIAM:
        return 1.0
    factor = CPU_DOWNSCALE_TARGET / min_dim
    factor = max(0.25, min(1.0, factor))  # never more than 4x
    return factor


def _downscale_for_nuclei(blue_channel, factor):
    """Downscale the nuclear channel. Returns (small_image, original_shape)."""
    if factor >= 0.999:
        return blue_channel, blue_channel.shape
    from skimage.transform import resize
    orig_shape = blue_channel.shape
    new_h = max(1, int(round(orig_shape[0] * factor)))
    new_w = max(1, int(round(orig_shape[1] * factor)))
    small = resize(blue_channel, (new_h, new_w), order=1,
                   preserve_range=True, anti_aliasing=True).astype(blue_channel.dtype)
    return small, orig_shape


def _upscale_mask(mask, orig_shape):
    """Scale a label mask back to the original size (nearest neighbour)."""
    if tuple(mask.shape) == tuple(orig_shape):
        return mask
    from skimage.transform import resize
    up = resize(mask, orig_shape, order=0, preserve_range=True,
                anti_aliasing=False).astype(np.int32)
    return up


# --- CellPose model cache ---
_CELLPOSE_MODEL = None
_CELLPOSE_MODEL_GPU = None
_CELLPOSE_IS_V4 = None  # detected once


def _cellpose_is_v4():
    """True for CellPose 4.x (CellposeModel/cpsam), False for 2.x."""
    global _CELLPOSE_IS_V4
    if _CELLPOSE_IS_V4 is None:
        from cellpose import models
        # models.Cellpose only exists in 2.x
        _CELLPOSE_IS_V4 = not hasattr(models, "Cellpose")
        try:
            import cellpose
            print(f"[CellPose] version {cellpose.version} "
                  f"({'API 4.x/cpsam' if _CELLPOSE_IS_V4 else 'API 2.x'})")
        except Exception:
            pass
    return _CELLPOSE_IS_V4


def _eval_cellpose(model, image, diameter, flow_threshold, cellprob_threshold):
    """Call model.eval on CellPose 2.x or 4.x and return only the masks.

    2.x needs channels=[0,0] and returns 4 values; 4.x (cpsam) takes a
    single-channel image directly and returns 3.
    """
    if _cellpose_is_v4():
        out = model.eval(image, diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold)
    else:
        out = model.eval(image, diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold,
                         channels=[0, 0])
    return out[0]  # masks always come first


def _get_cellpose_model():
    global _CELLPOSE_MODEL, _CELLPOSE_MODEL_GPU
    if _CELLPOSE_MODEL is None or _CELLPOSE_MODEL_GPU != USE_GPU:
        from cellpose import models
        print(f"[CellPose] loading nucleus model (GPU: {USE_GPU})...")
        if _cellpose_is_v4():
            # CellPose 4.x ships a single generalist model (cpsam)
            _CELLPOSE_MODEL = models.CellposeModel(gpu=USE_GPU)
            _CELLPOSE_MODEL_GPU = USE_GPU
            print("[CellPose] loaded cpsam (4.x)")
        else:
            # CellPose 2.x: modelo 'nuclei', fallback a 'cyto2'
            try:
                _CELLPOSE_MODEL = models.Cellpose(gpu=USE_GPU, model_type='nuclei')
                _CELLPOSE_MODEL_GPU = USE_GPU
                print("[CellPose] loaded 'nuclei' (2.x)")
            except Exception as e:
                print(f"[CellPose] could not load 'nuclei': {e}, trying 'cyto2'...")
                try:
                    _CELLPOSE_MODEL = models.Cellpose(gpu=USE_GPU, model_type='cyto2')
                    _CELLPOSE_MODEL_GPU = USE_GPU
                    print("[CellPose] loaded 'cyto2' (2.x)")
                except Exception as e2:
                    raise RuntimeError(
                        f"Could not load any CellPose model: {e2}\n"
                        "Check that the CellPose models are downloaded."
                    )
    return _CELLPOSE_MODEL


# --- image I/O ---
def load_image(filepath: str):
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    metadata = {"filepath": str(filepath), "format": ext}
    if ext == ".nd2":
        try:
            import nd2
            with nd2.ND2File(filepath) as f:
                img = f.asarray()
                metadata["nd2_metadata"] = str(f.metadata)
        except ImportError:
            raise ImportError("Instala nd2: pip install nd2")
    elif ext in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(str(filepath))
    else:
        raise ValueError(f"Formato no soportado: {ext}")
    metadata["original_shape"] = img.shape
    metadata["dtype"] = str(img.dtype)
    return img, metadata


def _standardize_channel(ch: np.ndarray, mode: str = "nuclei") -> np.ndarray:
    """Background subtraction + adaptive normalisation for one channel."""
    from skimage import exposure as sk_exp, filters as sk_f

    ch = ch.astype(np.float32)
    bit_max   = 65535.0 if ch.max() > 255 else 255.0
    sig_ratio = float(ch.max()) / bit_max
    frac_nz   = float((ch > 0).sum()) / ch.size

    print(f"  [{mode}] max={ch.max():.0f}/{bit_max:.0f} ({100*sig_ratio:.1f}%), p50={np.percentile(ch,50):.0f}")

    if ch.max() == 0:
        return np.zeros_like(ch, dtype=np.uint8)

    # nuclei: remove diffuse background and illumination gradients
    if mode == "nuclei":
        bg_s  = max(10, min(ch.shape) // 20)
        corr  = np.clip(ch - sk_f.gaussian(ch, sigma=bg_s), 0, None)
        # CLAHE brings back faint nuclei that a plain p1-p99 stretch would
        # flatten until CellPose cannot see them. A low clip limit keeps
        # noise from being amplified.
        corr_01 = corr / (corr.max() + 1e-6)
        corr    = sk_exp.equalize_adapthist(corr_01, clip_limit=0.01) * 255
        # gentle stretch on top, with wide percentiles so no signal is clipped
        lo, hi = np.percentile(corr, 0.5), np.percentile(corr, 99.8)
        if hi <= lo: hi = float(corr.max())
        result = np.clip((corr - lo) / (hi - lo + 1e-6) * 255, 0, 255)
        print(f"  [{mode}] background subtraction (sigma={bg_s}) + CLAHE, std={result.std():.1f}")

    else:  # amastigotes
        # Amastigotes are small bright spots; diffuse background must not count.
        # After background subtraction real parasites sit above ~20 and the
        # diffuse background below ~10, so an absolute floor works.
        bg_s = max(8, min(ch.shape) // 40)
        corr = np.clip(ch - sk_f.gaussian(ch, sigma=bg_s), 0, None)

        cvals = corr[corr > 0]
        if len(cvals) < 10:
            return np.zeros_like(ch, dtype=np.uint8)

        # threshold: an absolute minimum or a high percentile, whichever is
        # larger, to follow image brightness without picking up background
        p99   = float(np.percentile(corr, 99))
        p999  = float(np.percentile(corr, 99.9))
        noise_floor = max(15.0, p99 * 0.6)

        hi = max(p999, noise_floor + 10)
        # map noise_floor -> 30 and hi -> 255; below the floor -> 0
        result = np.clip((corr - noise_floor) / (hi - noise_floor + 1e-6) * 225 + 30, 0, 255)
        result[corr < noise_floor] = 0
        print(f"  [{mode}] background subtraction + absolute threshold: noise_floor={noise_floor:.1f}, hi={hi:.1f}")

    out = result.astype(np.uint8)
    print(f"  [{mode}] output: mean={out.mean():.1f}, std={out.std():.1f}, "
          f"pixels>0={100*(out>0).sum()/out.size:.3f}%")
    return out


def _estimate_nucleus_diameter(blue_channel: np.ndarray) -> int:
    """Measure the typical nucleus diameter in the image (any magnification)."""
    from skimage import filters as sk_f, morphology as sk_m, measure as sk_meas
    import math

    ch   = blue_channel.astype(np.float32)
    bg_s = max(10, min(ch.shape)//20)
    corr = np.clip(ch - sk_f.gaussian(ch, sigma=bg_s), 0, None)
    if corr.max() == 0:
        return 50

    lo, hi = np.percentile(corr, 1), np.percentile(corr, 99)
    norm = np.clip((corr-lo)/(hi-lo+1e-6)*255, 0, 255)
    sm   = sk_f.gaussian(norm, sigma=3)
    thr  = sk_f.threshold_otsu(sm)
    binary = _remove_small(sm > thr, 100)

    labels  = sk_meas.label(binary)
    regions = sk_meas.regionprops(labels)

    img_area = ch.shape[0] * ch.shape[1]
    valid = [r for r in regions if img_area*0.0003 <= r.area <= img_area*0.05]
    if len(valid) < 3:
        valid = [r for r in regions if 100 <= r.area <= img_area*0.1]
    if not valid:
        return 50

    diameters = [2*math.sqrt(r.area/math.pi) for r in valid]
    estimated = int(max(20, min(200, np.median(diameters))))
    print(f"[CellPose] nucleus diameter measured on the image: {estimated}px "
          f"(n={len(valid)}, range {min(diameters):.0f}-{max(diameters):.0f}px)")
    return estimated


def extract_channels(img: np.ndarray, blue_ch: int, red_ch: int,
                     amas_channel_indices: list = None,
                     meta: dict = None):
    """Extract and standardise the nuclear and parasite channels."""
    arr = img.astype(np.float32)
    if arr.ndim == 4:
        arr = arr.max(axis=0)

    def _get_ch(idx):
        if arr.ndim == 3:
            if arr.shape[0] <= 8:
                return arr[min(idx, arr.shape[0]-1)]
            else:
                return arr[:, :, min(idx, arr.shape[2]-1)]
        return arr.copy()

    # ND2: pick channels from their names
    fp = (meta or {}).get("filepath", "")
    if fp and fp.lower().endswith(".nd2"):
        try:
            import nd2 as nd2lib
            ab, ar = None, None
            with nd2lib.ND2File(fp) as f:
                for i, cm in enumerate(f.metadata.channels):
                    name = cm.channel.name.lower()
                    if any(x in name for x in ['blue','dapi','hoechst','qblue','cy5']):
                        ab = i
                    elif any(x in name for x in ['red','qred','cy3','tritc','texas',
                                                  'green','gfp','fitc','qgreen','alexa']):
                        ar = i
            if ab is not None and ar is not None:
                print(f"[Channels] ND2 auto: nuclei={ab}, amastigotes={ar}")
                blue_ch, red_ch = ab, ar
                amas_channel_indices = [ar]
        except Exception as e:
            print(f"[Channels] ND2 auto-detection failed: {e}")

    blue_raw = _get_ch(blue_ch)
    if amas_channel_indices and len(amas_channel_indices) > 0:
        channels = [_get_ch(i).astype(np.float32) for i in amas_channel_indices]
        red_raw  = np.max(np.stack(channels, axis=0), axis=0)
    else:
        red_raw = _get_ch(red_ch)

    print(f"[Channels] nuclei=channel {blue_ch}, amastigotes=channel {red_ch}")
    return _standardize_channel(blue_raw, "nuclei"), _standardize_channel(red_raw, "amastigotes")


# --- nucleus segmentation ---
def _count_nuclei_classic(blue_channel: np.ndarray, diameter: int) -> int:
    """Rough nucleus count with a classic watershed.

    Used as a reference to tell whether CellPose missed nuclei.
    """
    from skimage import filters as sk_f, morphology as sk_m, measure as sk_meas, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return 0

    sm  = sk_f.gaussian(ch, sigma=4)
    thr = sk_f.threshold_otsu(sm)
    binary = sm > (thr * 0.6)  # permissive threshold so faint nuclei are included
    min_area = int(np.pi * (diameter/2)**2 * 0.3)  # 30% of the expected nucleus area
    binary = _remove_small(binary, max(50, min_area))

    # watershed to split touching nuclei
    dist = ndimage.distance_transform_edt(binary)
    min_dist = max(10, int(diameter * 0.5))
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary)
    if len(coords) == 0:
        labels = sk_meas.label(binary)
        return int(labels.max())
    markers = np.zeros(binary.shape, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    ws = segmentation.watershed(-dist, markers, mask=binary)
    return int(ws.max())


def _count_nuclei_classic_safe(blue_channel, diameter):
    try:
        return _count_nuclei_classic(blue_channel, diameter)
    except Exception:
        return 0


def segment_nuclei_watershed(blue_channel: np.ndarray, diameter: int = None) -> np.ndarray:
    """Nucleus segmentation with a classic watershed.

    Handles faint nuclei that CellPose sometimes misses: nuclei are bright
    objects on a dark background.
    """
    from skimage import filters as sk_f, morphology as sk_m, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return np.zeros_like(ch, dtype=np.int32)

    if diameter is None:
        diameter = _estimate_nucleus_diameter(ch)

    # smooth, then a permissive threshold to keep faint nuclei
    sm  = sk_f.gaussian(ch, sigma=4)
    thr = sk_f.threshold_otsu(sm)
    binary = sm > (thr * 0.55)

    # drop small objects (amastigotes) and fill holes
    min_area = int(np.pi * (diameter/2)**2 * 0.25)
    binary = _remove_small(binary, max(100, min_area))
    binary = sk_m.closing(binary, sk_m.disk(3))
    binary = ndimage.binary_fill_holes(binary)

    # watershed to split touching nuclei
    dist = ndimage.distance_transform_edt(binary)
    # Smooth the distance transform in proportion to the diameter.
    # Nucleoli and internal texture create extra peaks that would split
    # one nucleus in two; without this, nucleolated nuclei are overcounted.
    dist = sk_f.gaussian(dist, sigma=max(1.0, diameter * 0.08))
    min_dist = max(4, int(diameter * 0.30))
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary,
                            exclude_border=False)

    if len(coords) == 0:
        labels = sk_m.label(binary) if hasattr(sk_m, 'label') else None
        from skimage import measure as _meas
        return _meas.label(binary).astype(np.int32)

    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    ws = segmentation.watershed(-dist, markers, mask=binary)

    print(f"[Watershed] {ws.max()} nuclei (diam={diameter}px, min_dist={min_dist})")
    return ws.astype(np.int32)


def _split_merged_nuclei(masks: np.ndarray, diameter: int) -> np.ndarray:
    """Split nuclei that CellPose merged into a single mask.

    Only labels that are clearly larger than one nucleus are touched; they
    are re-split by watershed on the distance transform.
    """
    from skimage import measure as sk_meas, segmentation, filters as sk_f
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    import math

    if masks.max() == 0:
        return masks

    expected_area = math.pi * (diameter / 2) ** 2
    regions = sk_meas.regionprops(masks)
    out = masks.copy().astype(np.int32)
    next_label = int(masks.max()) + 1
    split_count = 0

    for r in regions:
        # only act when the mask is clearly larger than one nucleus
        if r.area < expected_area * 1.6:
            continue
        sub = (masks[r.slice] == r.label)
        dist = ndi.distance_transform_edt(sub)
        # smoothing proportional to the diameter, so internal holes and
        # nucleoli do not split a nucleus
        dist = sk_f.gaussian(dist, sigma=max(1.0, diameter * 0.08))
        min_d = max(4, int(diameter * 0.30))
        coords = peak_local_max(dist, min_distance=min_d, labels=sub,
                                exclude_border=False)
        if len(coords) < 2:
            continue  # a single nucleus, leave it
        seeds = np.zeros(dist.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndi.label(seeds)
        ws = segmentation.watershed(-dist, markers, mask=sub)
        n_parts = int(ws.max())
        if n_parts < 2:
            continue
        # first piece keeps the label, the rest get new ones
        sl = r.slice
        region_out = out[sl]
        for p in range(1, n_parts + 1):
            piece = (ws == p) & sub
            if p == 1:
                region_out[piece] = r.label
            else:
                region_out[piece] = next_label
                next_label += 1
        out[sl] = region_out
        split_count += (n_parts - 1)

    if split_count:
        print(f"[Split] {split_count} merged nuclei split, total {out.max()}")
    return out


def _rescue_missed_nuclei(masks: np.ndarray, blue_channel: np.ndarray,
                          diameter: int) -> np.ndarray:
    """Recover nuclei that CellPose dropped for being diffuse or irregular.

    Looks for nuclear signal that was left unlabelled, checks its size and
    adds it as new nuclei. Large regions are split by watershed.
    """
    from skimage import filters as sk_f, measure as sk_meas, morphology as sk_m, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    import math

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return masks

    nuc_area = math.pi * (diameter / 2) ** 2

    # permissive mask of all nuclear signal, diffuse nuclei included
    thr = sk_f.threshold_otsu(ch)
    signal = ch > thr * 0.55
    signal = ndi.binary_fill_holes(signal)
    signal = _remove_small(signal, int(nuc_area * 0.4))

    # remove what CellPose already found (with a margin around the edges)
    already = sk_m.binary_dilation(masks > 0, sk_m.disk(2))
    missed  = signal & ~already
    missed  = _remove_small(missed, int(nuc_area * 0.5))

    if not missed.any():
        return masks

    out = masks.copy().astype(np.int32)
    next_label = int(masks.max()) + 1
    rescued = 0

    for r in sk_meas.regionprops(sk_meas.label(missed)):
        if r.area < nuc_area * 0.5:
            continue
        sub = np.zeros(ch.shape, bool)
        sub[r.slice][r.image] = True
        # large regions are split into several plausible nuclei
        if r.area > nuc_area * 1.6:
            dist = ndi.distance_transform_edt(sub)
            dist = sk_f.gaussian(dist, sigma=max(1.0, diameter * 0.08))
            coords = peak_local_max(dist, min_distance=max(4, int(diameter * 0.30)),
                                    labels=sub, exclude_border=False)
            if len(coords) >= 2:
                seeds = np.zeros(ch.shape, bool); seeds[tuple(coords.T)] = True
                mk, _ = ndi.label(seeds)
                ws = segmentation.watershed(-dist, mk, mask=sub)
                for p in range(1, int(ws.max()) + 1):
                    out[ws == p] = next_label; next_label += 1; rescued += 1
                continue
        out[sub] = next_label; next_label += 1; rescued += 1

    if rescued:
        print(f"[Rescue] {rescued} diffuse nuclei recovered, total {out.max()}")
    return out


def segment_nuclei(blue_channel: np.ndarray,
                   diameter=None,
                   flow_threshold: float = 0.4,
                   cellprob_threshold: float = 0.0) -> np.ndarray:
    """Segment nuclei with CellPose.

    CellPose 2.2.3 sometimes fails with diameter=None, so in that case the
    diameter is estimated from the image first.
    """
    
    if diameter is None:
        diameter = _estimate_nucleus_diameter(blue_channel)
        print(f"[CellPose] diameter=auto -> {diameter}px (measured on the image)")
    
    if blue_channel.max() == 0:
        print("[CellPose] warning: nuclear channel is empty")
        return np.zeros_like(blue_channel, dtype=np.int32)
    
    if blue_channel.std() < 5:
        print(f"[CellPose] warning: very low contrast in the nuclear channel (std={blue_channel.std():.1f})")

    # fast CPU mode: classic watershed instead of cpsam
    if CPU_FAST_NUCLEI and not USE_GPU:
        print(f"[CPU-fast] segmenting nuclei with a classic watershed "
              f"(diameter={diameter}px) instead of cpsam")
        try:
            masks = segment_nuclei_watershed(blue_channel, diameter)
            num_nuclei = int(masks.max())
            # split clumps and rescue diffuse nuclei, as on the CellPose path
            try:
                masks = _split_merged_nuclei(masks, diameter)
                num_nuclei = int(masks.max())
            except Exception as e:
                print(f"[Split] failed: {e}, keeping unsplit masks")
            try:
                masks = _rescue_missed_nuclei(masks, blue_channel, diameter)
                num_nuclei = int(masks.max())
            except Exception as e:
                print(f"[Rescue] failed: {e}, keeping masks without rescue")
            print(f"[CPU-fast] {num_nuclei} nuclei")
            return masks.astype(np.int32)
        except Exception as e:
            print(f"[CPU-fast] watershed failed ({e}), falling back to CellPose...")
            # if it fails, fall through to the normal CellPose path below

    print(f"[CellPose] diameter={diameter}, flow_thr={flow_threshold}, cellprob_thr={cellprob_threshold}")
    
    model = _get_cellpose_model()

    # CPU speed-up: run CellPose on a downscaled copy, diameter scaled too
    # (factor is 1.0 on GPU or for small images)
    ds_factor = _compute_downscale_factor(blue_channel.shape, diameter, USE_GPU)
    if ds_factor < 0.999:
        seg_input, _orig_shape = _downscale_for_nuclei(blue_channel, ds_factor)
        # scaled diameter, floor of 16 px (cpsam is fine from ~15 px)
        seg_diameter = max(16, int(round(diameter * ds_factor)))
        print(f"[CPU-speedup] nuclei at {ds_factor:.2f}x: "
              f"{blue_channel.shape} -> {seg_input.shape}, diameter {diameter} -> {seg_diameter}px")
    else:
        seg_input, _orig_shape = blue_channel, blue_channel.shape
        seg_diameter = diameter

    try:
        # first pass with the given parameters
        masks = _eval_cellpose(model, seg_input, seg_diameter,
                               flow_threshold, cellprob_threshold)
        num_nuclei = int(masks.max())
        print(f"[CellPose] pass 1 (cellprob={cellprob_threshold}, flow={flow_threshold}): {num_nuclei} nuclei")

        # how many nuclei do we expect, from the classic count?
        # (on the downscaled image too, so both numbers are comparable)
        expected = _count_nuclei_classic_safe(seg_input, seg_diameter)
        print(f"[CellPose] expected nuclei (classic estimate): ~{expected}")

        # One adaptive retry: if CellPose finds fewer nuclei than expected, a
        # second pass with a lower cellprob picks up the faint ones.
        # (An earlier version tried -1, -2 and -3; one well-chosen pass gets
        #  almost the same nuclei for a fraction of the CPU time.)
        if expected > 0 and num_nuclei < expected * 0.90:
            retry_thr = min(-2.0, cellprob_threshold - 1.0)
            print(f"[CellPose] low count ({num_nuclei}<{int(expected*0.90)}), "
                  f"retrying once with cellprob={retry_thr}...")
            try:
                m2 = _eval_cellpose(model, seg_input, seg_diameter,
                                    flow_threshold, retry_thr)
                n2 = int(m2.max())
                print(f"[CellPose]   -> {n2} nuclei")
                if n2 > num_nuclei and n2 <= max(expected * 1.5, num_nuclei + 1):
                    masks, num_nuclei = m2, n2
            except Exception as e:
                print(f"[CellPose]   retry failed: {e}")

        # back to full resolution before split/rescue, which work on the
        # full-resolution nuclear channel
        if ds_factor < 0.999:
            masks = _upscale_mask(masks, blue_channel.shape)
            num_nuclei = int(masks.max())
            print(f"[CPU-speedup] mask rescaled to {blue_channel.shape}: {num_nuclei} nuclei")

        # watershed fallback only if we are still far below the expected count
        # (expected count recomputed at full resolution)
        expected_full = _count_nuclei_classic_safe(blue_channel, diameter)
        if expected_full > 0 and num_nuclei < expected_full * 0.80:
            print(f"[CellPose] still low ({num_nuclei}), trying watershed...")
            try:
                ws_masks = segment_nuclei_watershed(blue_channel, diameter)
                num_ws = int(ws_masks.max())
                print(f"[Watershed] {num_ws} nuclei vs CellPose {num_nuclei}")
                if num_ws > num_nuclei:
                    masks = ws_masks
                    num_nuclei = num_ws
                    print("[CellPose] using the watershed result (better for faint nuclei)")
            except Exception as e:
                print(f"[Watershed] failed: {e}, keeping CellPose")

        # split clumps that ended up as a single mask
        try:
            masks = _split_merged_nuclei(masks, diameter)
            num_nuclei = int(masks.max())
        except Exception as e:
            print(f"[Split] failed: {e}, keeping unsplit masks")

        # rescue diffuse or irregular nuclei that CellPose dropped for their
        # shape (a lower cellprob does not bring them back)
        try:
            masks = _rescue_missed_nuclei(masks, blue_channel, diameter)
            num_nuclei = int(masks.max())
        except Exception as e:
            print(f"[Rescue] failed: {e}, keeping masks without rescue")

        print(f"[CellPose] {num_nuclei} nuclei")
        
        if num_nuclei == 0:
            print("[CellPose] warning: no nuclei detected")
            print("           Sugerencias:")
            print("           - try a lower cellprob_threshold (-1.0, -2.0)")
            print("           - adjust diameter (40-100 depending on nucleus size)")
        
        return masks.astype(np.int32)
        
    except Exception as e:
        print(f"[CellPose] segmentation error: {e}")
        raise


# --- amastigote detection with CellPose ---
_CELLPOSE_CYTO_MODEL = None
_CELLPOSE_CYTO_GPU   = None

def _get_cellpose_cyto_model():
    """Load (and cache) the CellPose model used for amastigotes."""
    global _CELLPOSE_CYTO_MODEL, _CELLPOSE_CYTO_GPU
    if _CELLPOSE_CYTO_MODEL is None or _CELLPOSE_CYTO_GPU != USE_GPU:
        from cellpose import models
        print(f"[CellPose-Amas] loading model (GPU: {USE_GPU})...")
        if _cellpose_is_v4():
            # 4.x: the single cpsam model handles nuclei and amastigotes
            _CELLPOSE_CYTO_MODEL = models.CellposeModel(gpu=USE_GPU)
            _CELLPOSE_CYTO_GPU   = USE_GPU
            print("[CellPose-Amas] loaded cpsam (4.x)")
        else:
            for mtype in ("cyto3", "cyto2", "cyto"):
                try:
                    _CELLPOSE_CYTO_MODEL = models.Cellpose(gpu=USE_GPU, model_type=mtype)
                    _CELLPOSE_CYTO_GPU   = USE_GPU
                    print(f"[CellPose-Amas] loaded '{mtype}'")
                    break
                except Exception as e:
                    print(f"[CellPose-Amas] '{mtype}' not available: {e}")
        if _CELLPOSE_CYTO_MODEL is None:
            raise RuntimeError("Could not load any CellPose model for amastigotes")
    return _CELLPOSE_CYTO_MODEL


def detect_amastigotes_cellpose(amas_channel: np.ndarray,
                                 min_size: int = 2,
                                 max_size: int = 150,
                                 expected_diameter: int = None) -> tuple:
    """Detect amastigotes with CellPose.

    Works best when parasites are well separated: it filters by shape, not
    only by brightness, so there are fewer false positives. Returns
    (sparse_mask, dense_mask, dense_analysis) like detect_amastigotes, or
    None if nothing was found (the caller then falls back to the classic
    method).
    """
    from skimage import measure
    import numpy as np

    arr = amas_channel.astype(np.float32)
    mean_val = float(arr.mean())
    std_val  = float(arr.std())

    if expected_diameter is None:
        expected_diameter = max(5, min(25, int(np.sqrt(max(max_size, 10)) * 1.2)))

    print(f"[CellPose-Amas] diameter={expected_diameter}, mean={mean_val:.1f}, std={std_val:.1f}")

    if std_val < 1.0 or arr.max() == 0:
        print("[CellPose-Amas] empty channel, skipping")
        return None

    try:
        model  = _get_cellpose_cyto_model()
        masks  = _eval_cellpose(model, amas_channel, expected_diameter,
                                flow_threshold=0.6, cellprob_threshold=-2.0)
    except Exception as e:
        print(f"[CellPose-Amas] Error: {e}")
        return None

    n_detected = int(masks.max())
    print(f"[CellPose-Amas] {n_detected} objects detected")
    if n_detected == 0:
        return None

    props = measure.regionprops(masks)
    keep  = np.zeros(masks.max() + 1, dtype=bool)
    dense_mask = np.zeros_like(masks, dtype=bool)
    dense_regions_list = []

    for r in props:
        area = int(r.area)
        if min_size <= area <= max_size:
            keep[r.label] = True
        elif area > max_size:
            dense_mask[masks == r.label] = True
            dense_regions_list.append({
                'label': int(r.label), 'area': area,
                'centroid': r.centroid, 'bbox': r.bbox,
                'intensity_mean': float(getattr(r, 'intensity_mean', 0)),
                'intensity_max':  float(getattr(r, 'intensity_max',  0)),
            })

    sparse_mask = np.where(keep[masks], masks, 0).astype(np.int32)
    n_sparse = int(sparse_mask.max())
    print(f"[CellPose-Amas] {n_sparse} sparse, {len(dense_regions_list)} dense")

    if n_sparse == 0 and not dense_regions_list:
        return None

    areas   = [float(r.area) for r in props if min_size <= int(r.area) <= max_size]
    avg_area = float(np.median(areas)) if len(areas) >= 5 else None
    cv       = float(np.std(areas) / avg_area) if avg_area else None
    cand_da  = max(int(max_size) * 3, 600)
    true_da  = max(int(avg_area * 110) if avg_area else 1600, 1600)
    conf     = 'high' if len(areas) >= 20 else ('medium' if len(areas) >= 5 else 'low')

    dense_analysis = {
        'regions': dense_regions_list,
        'total_dense_area': int(dense_mask.sum()),
        'red_channel_mean': mean_val, 'red_channel_std': std_val,
        'threshold': 0.0, 'red_channel': amas_channel,
        'detection_image': arr, 'binary_mask': masks > 0,
        'avg_sparse_amastigote_area': avg_area,
        'std_sparse_amastigote_area': float(np.std(areas)) if areas else None,
        'cv_sparse_amastigote': cv,
        'n_sparse_detected': len(areas), 'n_sparse_used_for_calibration': len(areas),
        'outliers_removed': 0, 'calibration_confidence': conf,
        'calibration_quality_score': min(10, len(areas) // 5),
        'candidate_dense_area': cand_da, 'true_dense_min_area': true_da,
        'v12_counting': True, 'detection_method': 'cellpose',
    }
    return sparse_mask, dense_mask, dense_analysis


def _should_use_cellpose_for_amastigotes(amas_channel: np.ndarray) -> bool:
    """Choose the detector in 'auto' mode.

    Dense signal (>12% of the area) -> classic method, more robust with
    masses; sparse signal -> CellPose, more precise with separate objects.
    """
    from skimage import filters
    arr = amas_channel.astype(np.float32)
    if arr.max() == 0 or arr.std() < 1:
        return False
    smoothed = filters.gaussian(arr, sigma=1.0)
    thr = float(np.percentile(smoothed, 95))
    signal_fraction = float((smoothed > thr).sum()) / float(smoothed.size)
    if signal_fraction > 0.12:
        print(f"[Auto mode] dense signal ({signal_fraction*100:.1f}%) -> classic")
        return False
    print(f"[Auto mode] sparse signal ({signal_fraction*100:.1f}%) -> CellPose")
    return True


# --- amastigote detection (classic) ---
def detect_amastigotes(red_channel: np.ndarray,
                       min_size: int = 2,
                       max_size: int = 150,
                       threshold_method: str = "otsu"):
    """Sensitive amastigote detection with a simple sparse/dense split.

    Small spots and small clusters go to sparse_mask and are counted later
    as discrete objects. Only components that are really dense masses go
    to dense_mask and are estimated from their area.
    """
    from skimage import filters, morphology, measure

    arr = red_channel.astype(np.float32)
    mean_val = float(arr.mean())
    std_val  = float(arr.std())
    max_val  = float(arr.max())

    print(f"[Detection] red channel: mean={mean_val:.1f}, std={std_val:.1f}, max={max_val:.1f}")

    def _empty_analysis(thresh=0.0):
        empty = np.zeros_like(red_channel, dtype=np.int32)
        dense_analysis = {
            'regions': [],
            'total_dense_area': 0,
            'red_channel_mean': mean_val,
            'red_channel_std': std_val,
            'threshold': float(thresh),
            'red_channel': red_channel,
            'detection_image': arr,
            'binary_mask': np.zeros_like(red_channel, dtype=bool),
            'avg_sparse_amastigote_area': None,
            'std_sparse_amastigote_area': None,
            'cv_sparse_amastigote': None,
            'n_sparse_detected': 0,
            'n_sparse_used_for_calibration': 0,
            'calibration_confidence': 'none',
            'calibration_quality_score': 0,
            'outliers_removed': 0,
            'candidate_dense_area': max(int(max_size) * 3, 600),
            'true_dense_min_area': 1600,
            'v12_counting': True,
        }
        return empty, np.zeros_like(red_channel, dtype=bool), dense_analysis

    if max_val <= 0 or std_val < 0.5:
        return _empty_analysis(0.0)

    # quick top-hat style enhancement of spots over the background: keeps
    # more amastigotes than a plain threshold without merging them
    sigma_bg = max(3, min(10, int(min(arr.shape) * 0.006)))
    bg = filters.gaussian(arr, sigma=sigma_bg, preserve_range=True)
    enhanced = arr - bg
    enhanced[enhanced < 0] = 0
    if float(enhanced.max()) < 2:
        enhanced = arr.copy()

    smoothed = filters.gaussian(enhanced, sigma=0.55, preserve_range=True)
    vals = smoothed[smoothed > 0]
    if vals.size < 10:
        return _empty_analysis(0.0)

    def _safe_threshold(fn, fallback):
        try:
            v = float(fn(smoothed))
            if np.isfinite(v) and 0 < v < float(smoothed.max()):
                return v
        except Exception:
            pass
        return float(fallback)

    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    p99 = float(np.percentile(vals, 99))
    stat = float(vals.mean() + 2.4 * vals.std())
    thresh_otsu = _safe_threshold(filters.threshold_otsu, stat)
    thresh_tri  = _safe_threshold(filters.threshold_triangle, stat)
    thresh_yen  = _safe_threshold(filters.threshold_yen, stat)

    method_map = {"otsu": thresh_otsu, "triangle": thresh_tri, "yen": thresh_yen}
    method_thr = method_map.get(threshold_method, thresh_otsu)

    # sensitive, but with a floor so faint halo and noise stay out
    # The floor and factor were lowered to catch medium-brightness
    # amastigotes (those mixed with some blue fluorescence, whose red
    # signal is weaker). With floor = p95 * 0.50 and otsu * 0.78 the
    # threshold was anchored to the brightest parasites and ~15% were lost.
    # remove_small_objects below still removes the noise, so the lower
    # threshold is safe.
    floor = max(p95 * 0.35, float(smoothed.max()) * 0.035, 1.0)
    thresh_final = max(method_thr * 0.65, floor)
    thresh_final = min(thresh_final, max(p99 * 0.92, floor))

    print(f"[Detection] candidates: stat={stat:.1f}, otsu={thresh_otsu:.1f}, triangle={thresh_tri:.1f}, yen={thresh_yen:.1f}, p95={p95:.1f}, p99={p99:.1f}")
    print(f"[Detection] final threshold: {thresh_final:.1f}")

    binary = smoothed > thresh_final
    binary = morphology.remove_small_objects(binary, min_size=max(1, int(min_size)))

    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels, intensity_image=smoothed)
    if not regions:
        return _empty_analysis(thresh_final)

    # calibration estimates the single-parasite size; it does not count
    small_upper = min(90, max(35, int(max_size * 0.65)))
    candidate_areas = []
    for r in regions:
        area = float(r.area)
        if max(1, int(min_size)) <= area <= small_upper:
            ecc = getattr(r, 'eccentricity', 0.0)
            if ecc <= 0.97:
                candidate_areas.append(area)

    avg_area = None
    std_dev = None
    cv = None
    filtered_areas = []
    n_total = len(candidate_areas)

    if n_total >= 10:
        areas = np.array(candidate_areas, dtype=float)
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        lower = max(float(min_size), q1 - 1.5 * iqr) if iqr > 0 else max(float(min_size), q1 * 0.7)
        upper = min(float(small_upper), q3 + 1.5 * iqr) if iqr > 0 else min(float(small_upper), q3 * 1.8 + 1)
        filtered = areas[(areas >= lower) & (areas <= upper)]
        if filtered.size >= 10:
            avg_area = float(np.median(filtered))
            std_dev = float(np.std(filtered))
            cv = std_dev / avg_area if avg_area > 0 else None
            # Smaller sizes than before are accepted so detection is not lost,
            # but a tiny value must not inflate the estimate for dense masses.
            if avg_area < 4.5 or (cv is not None and cv > 1.05):
                avg_area = None
                std_dev = None
                cv = None
            else:
                filtered_areas = filtered.tolist()

    n_filtered = len(filtered_areas)
    n_outliers = n_total - n_filtered

    confidence_score = 0
    if avg_area is not None:
        if n_filtered >= 45:
            confidence_score += 4
        elif n_filtered >= 25:
            confidence_score += 3
        elif n_filtered >= 10:
            confidence_score += 2
        if cv is not None:
            if cv < 0.35:
                confidence_score += 3
            elif cv < 0.65:
                confidence_score += 2
            elif cv < 0.90:
                confidence_score += 1
        if 5 <= avg_area <= 65:
            confidence_score += 3
        elif 4.5 <= avg_area <= 90:
            confidence_score += 1

    if confidence_score >= 8:
        confidence_level = 'high'
    elif confidence_score >= 5:
        confidence_level = 'medium'
    else:
        confidence_level = 'very_low'
        avg_area = None

    unit_for_thresholds = float(avg_area) if avg_area is not None else 12.0

    # only clearly large components go to dense_mask; the cut-off is high
    # enough that sparse spots (e.g. at 24 hpi) never count as masses
    candidate_dense_area = max(int(max_size) * 3, int(unit_for_thresholds * 38), 600)
    true_dense_min_area  = max(int(unit_for_thresholds * 110), 1600)

    sparse_keep = np.zeros(max(labels.max() + 1, 1), dtype=bool)
    dense_mask = np.zeros_like(binary, dtype=bool)
    dense_regions_list = []

    for r in regions:
        area = int(r.area)
        if area >= candidate_dense_area:
            dense_mask[labels == r.label] = True
            dense_regions_list.append({
                'label': int(r.label),
                'area': area,
                'centroid': r.centroid,
                'bbox': r.bbox,
                'intensity_mean': float(r.intensity_mean),
                'intensity_max': float(r.intensity_max),
            })
        elif area >= int(min_size):
            sparse_keep[r.label] = True

    sparse_mask = np.where(sparse_keep[labels], labels, 0).astype(np.int32)
    dense_mask = dense_mask.astype(bool)

    if avg_area is not None:
        print(f"[Calibration] accepted: {avg_area:.1f}px^2, n={n_filtered}, cv={cv if cv is not None else 0:.2f}, confidence={confidence_level}")
    else:
        print(f"[Calibration] not used for masses (n={n_total}); discrete counting only")

    print(f"[Detection] sparse labels={int(sparse_mask.max())}, dense candidates={len(dense_regions_list)}, dense_area>={candidate_dense_area}px^2, true mass>={true_dense_min_area}px^2")

    dense_analysis = {
        'regions': dense_regions_list,
        'total_dense_area': int(dense_mask.sum()),
        'red_channel_mean': mean_val,
        'red_channel_std': std_val,
        'threshold': float(thresh_final),
        'red_channel': red_channel,
        'detection_image': smoothed,
        'binary_mask': binary,
        'avg_sparse_amastigote_area': avg_area,
        'std_sparse_amastigote_area': std_dev,
        'cv_sparse_amastigote': cv,
        'n_sparse_detected': n_total,
        'n_sparse_used_for_calibration': n_filtered,
        'outliers_removed': n_outliers,
        'calibration_confidence': confidence_level,
        'calibration_quality_score': confidence_score,
        'candidate_dense_area': int(candidate_dense_area),
        'true_dense_min_area': int(true_dense_min_area),
        'v12_counting': True,
    }

    return sparse_mask, dense_mask, dense_analysis


"""Watershed splitting of amastigote clusters for calibration."""


# --- watershed to split clusters ---
def separate_amastigote_clusters(sparse_mask: np.ndarray, 
                                 red_channel: np.ndarray,
                                 expected_size: float = 13) -> np.ndarray:
    """Split clusters of touching amastigotes with a watershed.

    Conservative: only objects clearly larger than one amastigote are
    split, so small noise is left alone (and it is faster).
    """
    from skimage import measure, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage
    
    if sparse_mask.max() == 0:
        return sparse_mask
    
    regions = measure.regionprops(sparse_mask)
    separated_mask = np.zeros_like(sparse_mask, dtype=np.int32)
    label_counter = 1
    cluster_trigger = max(expected_size * 2.8, 22)
    
    for region in regions:
        if region.area > cluster_trigger:
            bbox = region.bbox
            sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
            roi_mask = sparse_mask[sl] == region.label
            roi_signal = red_channel[sl]
            
            distance = ndimage.distance_transform_edt(roi_mask)
            min_distance = max(2, int(np.sqrt(max(expected_size, 4))))

            # signal maxima first; distance transform if there are not enough
            vals = roi_signal[roi_mask]
            if vals.size:
                thr = max(float(np.percentile(vals, 65)), float(vals.max()) * 0.35)
            else:
                thr = 0
            try:
                coords = peak_local_max(
                    roi_signal,
                    min_distance=min_distance,
                    threshold_abs=thr,
                    labels=roi_mask,
                    exclude_border=False,
                    num_peaks=max(1, int(round(region.area / max(expected_size, 4)))) + 3
                )
            except Exception:
                coords = np.empty((0, 2), dtype=int)

            if len(coords) <= 1:
                try:
                    coords = peak_local_max(
                        distance,
                        min_distance=min_distance,
                        labels=roi_mask,
                        exclude_border=False,
                        num_peaks=max(1, int(round(region.area / max(expected_size, 4)))) + 3
                    )
                except Exception:
                    coords = np.empty((0, 2), dtype=int)
            
            if len(coords) > 1:
                markers = np.zeros_like(roi_mask, dtype=np.int32)
                for i, coord in enumerate(coords):
                    markers[coord[0], coord[1]] = i + 1
                
                try:
                    labels_roi = segmentation.watershed(-distance, markers, mask=roi_mask)
                    target = separated_mask[sl]
                    for i in range(1, labels_roi.max() + 1):
                        mask_i = labels_roi == i
                        if mask_i.sum() >= 1:
                            target[mask_i] = label_counter
                            label_counter += 1
                except Exception:
                    separated_mask[sparse_mask == region.label] = label_counter
                    label_counter += 1
            else:
                separated_mask[sparse_mask == region.label] = label_counter
                label_counter += 1
        else:
            separated_mask[sparse_mask == region.label] = label_counter
            label_counter += 1
    
    return separated_mask


# --- infection criterion ---
def compute_infection(nucleus_mask: np.ndarray,
                      sparse_mask: np.ndarray,
                      dense_mask: np.ndarray,
                      dense_analysis: dict,
                      dilation_radius: int = 100,
                      min_amastigotes: int = 1,
                      avg_amastigote_size: int = 15) -> dict:
    """Assign parasites to cells and count them.

    - Each red object is assigned to the nearest nucleus (KD-tree).
    - Isolated spots count as one amastigote each.
    - Non-dense clusters are counted from local maxima, capped so they do
      not jump artificially above 50.
    - Only real dense masses (very large and covering a relevant part of
      the cell territory) are estimated from the calibrated area.
    """
    from skimage import measure
    from skimage.feature import peak_local_max
    from scipy.spatial import cKDTree

    n_nuclei = int(nucleus_mask.max())
    if n_nuclei == 0:
        return {
            "total_cells": 0, "infected_cells": 0, "infection_pct": 0.0,
            "per_cell": [], "infected_mask": np.zeros_like(nucleus_mask, dtype=bool),
            "cytosol_mask": nucleus_mask.copy(), "calibration": {}
        }

    nuc_props = measure.regionprops(nucleus_mask)
    if not nuc_props:
        return {
            "total_cells": 0, "infected_cells": 0, "infection_pct": 0.0,
            "per_cell": [], "infected_mask": np.zeros_like(nucleus_mask, dtype=bool),
            "cytosol_mask": nucleus_mask.copy(), "calibration": {}
        }

    nuc_labels = np.array([int(p.label) for p in nuc_props], dtype=np.int32)
    nuc_centroid_arr = np.array([p.centroid for p in nuc_props], dtype=np.float32)
    nuc_centroids = {int(p.label): p.centroid for p in nuc_props}
    nuc_areas = {int(p.label): int(p.area) for p in nuc_props}
    tree = cKDTree(nuc_centroid_arr)

    # effective calibration: only for clusters and masses, never for spots
    detected_size = dense_analysis.get('avg_sparse_amastigote_area')
    confidence = dense_analysis.get('calibration_confidence', 'none')
    n_valid = int(dense_analysis.get('n_sparse_used_for_calibration', 0) or 0)
    quality_score = int(dense_analysis.get('calibration_quality_score', 0) or 0)

    if detected_size is not None and confidence in ['high', 'medium'] and n_valid >= 10:
        effective_amas_size = float(detected_size)
        calibration_source = "auto"
        calibration_confidence = confidence
    elif avg_amastigote_size > 0 and 5 <= avg_amastigote_size <= 150:
        effective_amas_size = float(avg_amastigote_size)
        calibration_source = "manual"
        calibration_confidence = "user"
    else:
        effective_amas_size = 12.0
        calibration_source = "default"
        calibration_confidence = "fallback"

    # guard against overcounting: never divide by a tiny unit size
    sparse_unit = max(float(effective_amas_size), 8.0)
    dense_unit = max(float(effective_amas_size) * 1.45, 16.0)
    candidate_dense_area = int(dense_analysis.get('candidate_dense_area') or max(600, sparse_unit * 38))
    true_dense_min_area = int(dense_analysis.get('true_dense_min_area') or max(1600, sparse_unit * 110))

    print(f"[Count] calibration {calibration_source}: unit={effective_amas_size:.1f}px^2; sparse_unit={sparse_unit:.1f}; dense_unit={dense_unit:.1f}")

    # rough territory per cell, used to validate dense masses
    if len(nuc_props) > 1:
        nn_dists, _ = tree.query(nuc_centroid_arr, k=2)
        nearest_neighbor = nn_dists[:, 1]
    else:
        nearest_neighbor = np.array([2 * dilation_radius], dtype=float)

    territory_area = np.zeros(n_nuclei + 1, dtype=float)
    # Maximum assignment radius per nucleus: nucleus edge + cytoplasm margin.
    # An amastigote further away is "free" and is not assigned.
    max_assign_radius = np.zeros(n_nuclei + 1, dtype=float)
    for i, lbl in enumerate(nuc_labels):
        local_r = min(float(dilation_radius), max(18.0, float(nearest_neighbor[i]) * 0.55))
        territory_area[int(lbl)] = float(nuc_areas.get(int(lbl), 0)) + np.pi * local_r * local_r
        # nucleus radius from its real area
        nuc_area = float(nuc_areas.get(int(lbl), 200))
        nuc_radius = np.sqrt(nuc_area / np.pi)
        # Cytoplasm margin: how far from the nucleus amastigotes are assigned.
        # In T. cruzi the amastigotes fill the cytoplasm as a ring that reaches
        # ~1.5x the nucleus radius. The old margin (nearest_neighbor * 0.6) was
        # too short: it missed that ring in clearly infected cells and marked
        # them as uninfected.
        #
        # Current rule:
        #   - biological floor: 1.5 x nucleus radius (typical cytoplasm)
        #   - neighbour cap: nearest_neighbor * 0.75 (stay out of adjacent cells)
        #   - the final margin keeps the floor even when a neighbour is close,
        #     but never exceeds dilation_radius.
        cito_floor = nuc_radius * 1.5
        nbr_limit  = float(nearest_neighbor[i]) * 0.75
        cito_margin = max(cito_floor, nbr_limit)
        cito_margin = min(cito_margin, float(dilation_radius))
        max_assign_radius[int(lbl)] = nuc_radius + cito_margin

    def _owner_from_region(reg) -> int:
        cy, cx = reg.centroid
        dist, idx = tree.query([float(cy), float(cx)], k=1)
        owner = int(nuc_labels[int(idx)])
        # only assign if the amastigote is within that cell's radius;
        # anything further away is free background signal
        if float(dist) <= float(max_assign_radius[owner]):
            return owner
        return 0

    det_img = dense_analysis.get('detection_image')
    if det_img is None:
        det_img = dense_analysis.get('red_channel')
    if det_img is None:
        det_img = np.zeros_like(nucleus_mask, dtype=np.float32)
    det_img = det_img.astype(np.float32)

    binary_signal = dense_analysis.get('binary_mask')
    if binary_signal is None:
        thresh_used = float(dense_analysis.get('threshold', 0) or 0)
        binary_signal = det_img > thresh_used
    else:
        binary_signal = binary_signal.astype(bool)

    def _local_peaks(label_img: np.ndarray, reg, min_distance: int = 4,
                     percentile: float = 78, max_peaks: int = 80) -> int:
        """Bright local maxima, with conservative parameters.

        Only a fallback: with min_distance=3 and low percentiles, maxima
        used to multiply inside a single amastigote. The main count comes
        from bright cores separated by a high local threshold.
        """
        bbox = reg.bbox
        sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
        local_mask = label_img[sl] == reg.label
        if not local_mask.any():
            return 0
        roi = det_img[sl]
        vals = roi[local_mask]
        if vals.size == 0 or float(vals.max()) <= 0:
            return 0

        # high threshold so one spot does not give several peaks
        thr = max(float(np.percentile(vals, percentile)), float(vals.max()) * 0.45)
        try:
            coords = peak_local_max(
                roi,
                min_distance=max(4, int(min_distance)),
                threshold_abs=thr,
                labels=local_mask,
                exclude_border=False,
                num_peaks=max_peaks,
            )
            return int(len(coords))
        except Exception:
            return 0

    def _count_bright_cores(label_img: np.ndarray, reg) -> int:
        """Count visible amastigotes as separate bright centres.

        For discrete signal, area / mean size is not used: halo and smoothing
        can merge 2 spots into one large component, which then counted as
        12-14. Instead each component is re-thresholded with a high local
        threshold, so two touching spots stay two.
        """
        from skimage import measure, morphology

        bbox = reg.bbox
        sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
        local_mask = label_img[sl] == reg.label
        if not local_mask.any():
            return 0

        roi = det_img[sl]
        vals = roi[local_mask]
        if vals.size == 0 or float(vals.max()) <= 0:
            return 0

        vmax = float(vals.max())
        # High local threshold: a single spot with a halo could otherwise split
        # into two cores and double the count. Keep only the really bright
        # centre and require reasonably solid cores.
        q86 = float(np.percentile(vals, 86))
        q92 = float(np.percentile(vals, 92))
        thr_core = max(q86, q92 * 0.92, vmax * 0.62)
        core = (roi >= thr_core) & local_mask

        try:
            from skimage import morphology as _sk_morph
            core = _sk_morph.remove_small_objects(core, min_size=3)
        except Exception:
            pass

        core_labels = measure.label(core, connectivity=2)
        core_props = measure.regionprops(core_labels, intensity_image=roi)
        if not core_props:
            return 0

        # Drop tiny cores: one amastigote can have 1-2 maximal pixels
        # separated by noise. Only cores with enough area or clearly
        # dominant brightness count as independent centres.
        valid = []
        for cp in core_props:
            imax = float(getattr(cp, 'intensity_max', 0.0))
            if cp.area >= 3 or (cp.area >= 2 and imax >= vmax * 0.88):
                valid.append(cp)

        return int(len(valid))

    def _count_discrete_component(label_img: np.ndarray, reg) -> int:
        """Count sparse spots and clusters.

        - small spot: 1
        - discrete cluster: number of separate bright centres
        - area is never divided by the amastigote size for discrete signal
          (that was what turned 2 spots into 14)
        - only true dense masses, validated further down, are estimated from
          area and can reach the 15-50 or >50 ranges
        """
        area = float(reg.area)
        # Raise the "single amastigote" area. At 24 hpi the discrete spots
        # have a halo, which used to make one isolated spot count as 2. Now
        # every small component counts as 1 unless it is clearly a large
        # cluster.
        min_single_area = max(3.4 * sparse_unit, 36.0)

        h = max(1, int(reg.bbox[2] - reg.bbox[0]))
        w = max(1, int(reg.bbox[3] - reg.bbox[1]))
        major_axis = max(h, w)
        minor_axis = min(h, w)
        diameter_guess = max(5.0, float(np.sqrt(max(sparse_unit, 8.0))) * 1.75)

        if area <= min_single_area and major_axis <= diameter_guess * 1.75:
            return 1

        cores = _count_bright_cores(label_img, reg)
        peaks = 0
        # the peak-based fallback is stricter: only for objects large or
        # elongated enough to hold several amastigotes
        if area > min_single_area * 1.05 or major_axis > diameter_guess * 1.8:
            peaks = _local_peaks(
                label_img, reg,
                min_distance=max(5, int(round(diameter_guess * 0.9))),
                percentile=86,
                max_peaks=80,
            )
        visible_count = int(max(cores, peaks))

        if visible_count <= 0:
            if area < candidate_dense_area:
                return 1 if area < min_single_area * 2.2 else 2
            return 3

        # Strict geometric cap. In the 1-15 range it is better to undercount
        # by one than to turn 2 visible spots into 4. Real large clusters can
        # still reach 15-50 through the number of separate centres.
        max_by_length = max(1, int(np.ceil(major_axis / diameter_guess)))
        max_by_area = max(1, int(np.ceil(area / max(28.0, sparse_unit * 3.2))))
        if minor_axis > diameter_guess * 1.7 and area > min_single_area * 2.5:
            # compact 2-D cluster: allow a bit more than its length
            max_by_shape = max_by_area
        else:
            max_by_shape = max_by_length
        geom_cap = max(1, min(max_by_shape, max_by_area))

        if area < candidate_dense_area:
            return max(1, min(visible_count, geom_cap, 15))

        return max(1, min(visible_count, geom_cap, 30))

    counts_total = np.zeros(n_nuclei + 1, dtype=int)
    counts_sparse = np.zeros(n_nuclei + 1, dtype=int)
    counts_cluster = np.zeros(n_nuclei + 1, dtype=int)
    counts_dense = np.zeros(n_nuclei + 1, dtype=int)
    dense_fraction_by_cell = np.zeros(n_nuclei + 1, dtype=float)

    # 1) discrete objects: assignment and protected count
    amas_props = measure.regionprops(sparse_mask, intensity_image=det_img)
    assigned_sparse = 0
    # For the second (contiguity) pass: amastigote label -> owning cell,
    # and the amastigotes left unassigned after the first pass.
    owner_of_amas = {}          # reg.label -> owner_id (assigned only)
    free_regions = []           # regions without an owner after pass 1
    for reg in amas_props:
        owner_id = _owner_from_region(reg)
        if owner_id == 0:
            free_regions.append(reg)   # candidate for pass 2
            continue  # free amastigote in the background, not assigned for now
        owner_of_amas[reg.label] = owner_id
        n_obj = _count_discrete_component(sparse_mask, reg)
        if n_obj <= 0:
            continue
        counts_sparse[owner_id] += n_obj
        counts_total[owner_id] += n_obj
        assigned_sparse += n_obj

    # 1b) second pass: contiguity
    # A free amastigote touching a cluster already assigned to cell X
    # belongs to the same infection (the parasites of one cell form a
    # connected cluster). This recovers the outer edge of the ring
    # without assigning parasites that are really isolated.
    # It relies on actual contact rather than on a fixed radius that would
    # need tuning for every image.
    if free_regions and owner_of_amas:
        from skimage import morphology as sk_morph
        from scipy import ndimage as sk_ndi

        # which cell owns each assigned amastigote pixel
        owner_pixels = np.zeros_like(sparse_mask, dtype=np.int32)
        for lbl_a, own in owner_of_amas.items():
            owner_pixels[sparse_mask == lbl_a] = own

        # Contact distance scales with the single-amastigote size, so it
        # adapts to each image. A factor of 3.0 joins parasites touching the
        # cluster (ring edge, ~1-1.5 diameters apart) but does not bridge to
        # really isolated ones further out, which would be false assignments.
        link_dist = max(3, int(round(np.sqrt(max(sparse_unit, 4.0) / np.pi) * 3.0)))
        struct = sk_morph.disk(link_dist)

        # Several iterations: a newly joined amastigote can in turn bridge to
        # another free one further out (a chain along the ring).
        changed = True
        passes = 0
        while changed and passes < 4 and free_regions:
            changed = False
            passes += 1
            still_free = []
            # dilate the owner map to detect contact
            owner_dil = sk_ndi.grey_dilation(owner_pixels, footprint=struct)
            for reg in free_regions:
                coords = reg.coords
                # cell labels touched around this free amastigote
                touched = owner_dil[coords[:, 0], coords[:, 1]]
                touched = touched[touched > 0]
                if touched.size > 0:
                    # owner = majority cell in the contact zone
                    vals, cnts = np.unique(touched, return_counts=True)
                    own = int(vals[int(np.argmax(cnts))])
                    owner_of_amas[reg.label] = own
                    owner_pixels[sparse_mask == reg.label] = own
                    n_obj = _count_discrete_component(sparse_mask, reg)
                    if n_obj > 0:
                        counts_sparse[own] += n_obj
                        counts_total[own] += n_obj
                        assigned_sparse += n_obj
                    changed = True
                else:
                    still_free.append(reg)
            free_regions = still_free

    # 1c) third pass: nearest nucleus, with a distance cap
    # Amastigotes still free after the contiguity pass usually sit in the
    # space between cells, clearly closer to ONE cell but just outside its
    # strict radius. They go to the nearest nucleus if they are within an
    # extended distance (1.6x that nucleus's assignment radius). Parasites
    # really lost in the background (further than that) stay unassigned,
    # which avoids false positives.
    if free_regions:
        EXTEND = 1.6   # how much the strict radius is extended in this last pass
        still_free = []
        for reg in free_regions:
            cy, cx = reg.centroid
            dist, idx = tree.query([float(cy), float(cx)], k=1)
            owner = int(nuc_labels[int(idx)])
            if float(dist) <= float(max_assign_radius[owner]) * EXTEND:
                n_obj = _count_discrete_component(sparse_mask, reg)
                if n_obj > 0:
                    counts_sparse[owner] += n_obj
                    counts_total[owner] += n_obj
                    assigned_sparse += n_obj
            else:
                still_free.append(reg)
        free_regions = still_free

    # 2) dense candidates: only very large masses are counted by area
    dense_labels = measure.label(dense_mask, connectivity=2)
    dense_props = measure.regionprops(dense_labels, intensity_image=det_img)
    true_dense_components = 0
    fallback_dense_as_cluster = 0
    assigned_dense = 0

    for reg in dense_props:
        owner_id = _owner_from_region(reg)
        if owner_id == 0:
            continue  # dense mass free in the background, not assigned
        bbox = reg.bbox
        sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
        local_mask = dense_labels[sl] == reg.label
        signal_px = int((binary_signal[sl] & local_mask).sum())
        signal_area = max(signal_px, int(reg.area))
        terr = max(1.0, float(territory_area[owner_id] if territory_area[owner_id] > 0 else np.pi * dilation_radius ** 2))
        dense_fraction = signal_px / terr
        dense_fraction_by_cell[owner_id] = max(dense_fraction_by_cell[owner_id], float(dense_fraction))

        peaks = _local_peaks(dense_labels, reg, min_distance=3, percentile=62, max_peaks=650)

        # true dense mass: clearly large and covering a relevant local fraction
        true_mass = (
            signal_area >= true_dense_min_area and
            (dense_fraction >= 0.08 or signal_area >= true_dense_min_area * 1.8) and
            (peaks >= 18 or signal_area >= true_dense_min_area * 2.2)
        )

        if true_mass:
            area_est = int(round(signal_area / dense_unit))
            # real masses may exceed 50, with a practical biological limit
            n_obj = max(15, min(max(area_est, peaks), 500))
            counts_dense[owner_id] += n_obj
            true_dense_components += 1
        else:
            # not a real mass: treat as a capped discrete cluster
            n_obj = _count_discrete_component(dense_labels, reg)
            counts_cluster[owner_id] += n_obj
            fallback_dense_as_cluster += 1

        counts_total[owner_id] += n_obj
        assigned_dense += n_obj


    cytosol_mask = nucleus_mask.astype(np.int32)
    per_cell = []
    infected_labels = set()

    for nuc_id in range(1, n_nuclei + 1):
        total_count = int(counts_total[nuc_id])
        sparse_count = int(counts_sparse[nuc_id])
        cluster_count = int(counts_cluster[nuc_id])
        dense_count = int(counts_dense[nuc_id])
        infected = total_count >= int(min_amastigotes)
        if infected:
            infected_labels.add(nuc_id)

        if total_count == 0:
            mode = "sparse"
            display = 0
        elif total_count <= 15:
            mode = "sparse" if dense_count == 0 else "low_dense"
            display = total_count
        elif total_count <= 50:
            mode = "sparse" if dense_count == 0 else "medium_dense"
            display = "15-50"
        else:
            # >50 only from a real dense mass or from >50 real components
            mode = "high_dense" if dense_count > 0 else "sparse"
            display = ">50"

        cy, cx = nuc_centroids.get(nuc_id, (0, 0))
        per_cell.append({
            "nucleus_id": nuc_id,
            "centroid_y": cy,
            "centroid_x": cx,
            "amastigotes_nearby": display,
            "amastigotes_count": total_count,
            "infected": infected,
            "area_px": int(nuc_areas.get(nuc_id, 0)),
            "mode": mode,
            "dense_fraction": float(dense_fraction_by_cell[nuc_id]),
            "sparse_count": sparse_count,
            "cluster_count": cluster_count,
            "dense_count": dense_count,
            "territory_area": int(round(territory_area[nuc_id])) if territory_area[nuc_id] > 0 else 0,
        })

    infected_mask = np.isin(nucleus_mask, list(infected_labels)) if infected_labels else np.zeros_like(nucleus_mask, dtype=bool)
    infected_cells = len(infected_labels)
    total_cells = len(per_cell)
    pct = round(infected_cells / total_cells * 100, 2) if total_cells > 0 else 0.0

    return {
        "total_cells": total_cells,
        "infected_cells": infected_cells,
        "infection_pct": pct,
        "per_cell": per_cell,
        "infected_mask": infected_mask,
        "cytosol_mask": cytosol_mask,
        "calibration": {
            "source": calibration_source,
            "confidence": calibration_confidence,
            "quality_score": quality_score,
            "amas_size_used": round(effective_amas_size, 2),
            "sparse_unit_used": round(sparse_unit, 2),
            "dense_unit_used": round(dense_unit, 2),
            "candidate_dense_area": int(candidate_dense_area),
            "true_dense_min_area": int(true_dense_min_area),
            "n_detected": dense_analysis.get('n_sparse_detected', 0),
            "n_used": n_valid,
            "n_outliers": dense_analysis.get('outliers_removed', 0),
            "cv": dense_analysis.get('cv_sparse_amastigote'),
            "std_dev": dense_analysis.get('std_sparse_amastigote_area'),
            "model": "sensitive_detection_nearest_assignment_realistic_counting_v11"
        }
    }


# --- overlay ---
def build_overlay(blue_ch: np.ndarray,
                  red_ch: np.ndarray,
                  nucleus_mask: np.ndarray,
                  sparse_mask: np.ndarray,
                  dense_mask: np.ndarray,
                  results: dict,
                  amas_color: str = "red") -> np.ndarray:
    from skimage import segmentation, morphology
    from scipy.ndimage import binary_dilation as bdil
    from PIL import Image, ImageDraw, ImageFont

    h, w = blue_ch.shape
    has_dense = any(c["mode"] in ["low_dense", "medium_dense", "high_dense"] 
                    for c in results["per_cell"])
    red_mult  = 2 if has_dense else 8

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = np.clip(blue_ch.astype(np.uint16) * 2, 0, 255).astype(np.uint8)
    # amastigote channel painted red or green depending on amas_color
    amas_signal = np.clip(red_ch.astype(np.uint16) * red_mult, 0, 255).astype(np.uint8)
    if amas_color == "green":
        rgb[:, :, 1] = amas_signal   # signal in the green channel
    else:
        rgb[:, :, 0] = amas_signal   # signal in the red channel

    infected_ids = {c["nucleus_id"] for c in results["per_cell"] if c["infected"]}
    healthy_ids  = {c["nucleus_id"] for c in results["per_cell"] if not c["infected"]}

    # orange outline for dense masses
    if dense_mask.any():
        db = bdil(segmentation.find_boundaries(dense_mask, mode='outer'),
                  structure=morphology.disk(max(2, min(h,w)//800)))
        rgb[db, 0] = 255; rgb[db, 1] = 140; rgb[db, 2] = 0

    # nucleus fill: yellow infected, green uninfected
    if infected_ids:
        m = np.isin(nucleus_mask, list(infected_ids))
        rgb[m, 0] = np.clip(rgb[m, 0].astype(int) + 180, 0, 255).astype(np.uint8)
        rgb[m, 1] = np.clip(rgb[m, 1].astype(int) + 180, 0, 255).astype(np.uint8)
        rgb[m, 2] = np.clip(rgb[m, 2].astype(int) -  80, 0, 255).astype(np.uint8)
    if healthy_ids:
        m = np.isin(nucleus_mask, list(healthy_ids))
        rgb[m, 0] = np.clip(rgb[m, 0].astype(int) -  50, 0, 255).astype(np.uint8)
        rgb[m, 1] = np.clip(rgb[m, 1].astype(int) + 150, 0, 255).astype(np.uint8)
        rgb[m, 2] = np.clip(rgb[m, 2].astype(int) -  50, 0, 255).astype(np.uint8)

    # white outline for every nucleus
    thick = max(2, min(h, w) // 600)
    tb = bdil(segmentation.find_boundaries(nucleus_mask, mode='outer'),
              structure=morphology.disk(thick))
    rgb[tb] = [255, 255, 255]

    # outline of individual amastigotes (colour depends on mode)
    if sparse_mask.max() > 0:
        ab = bdil(segmentation.find_boundaries(sparse_mask > 0, mode='outer'),
                  structure=morphology.disk(thick))
        if amas_color == "green":
            rgb[ab] = [0, 255, 0]      # bright green
        else:
            rgb[ab] = [255, 0, 255]    # Magenta

    img_pil = Image.fromarray(rgb)
    drw     = ImageDraw.Draw(img_pil)
    fsz     = max(18, min(h, w) // 70)
    try:
        font = ImageFont.truetype("arialbd.ttf", fsz)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", fsz)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsz)
            except Exception:
                font = ImageFont.load_default()

    for cell in results["per_cell"]:
        cy  = int(cell["centroid_y"])
        cx  = int(cell["centroid_x"])
        col = (255, 230, 0) if cell["infected"] else (0, 255, 180)
        lines = [str(cell["nucleus_id"])]
        if cell["infected"]:
            if cell["mode"] in ("low_dense", "sparse"):
                lines.append(str(cell["amastigotes_count"]))
            elif cell["mode"] == "medium_dense":
                lines.append("15-50")
            elif cell["mode"] == "high_dense":
                lines.append(">50")

        for i, line in enumerate(lines):
            try:
                bbox = drw.textbbox((0, 0), line, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            except Exception:
                tw, th = fsz * len(line) // 2, fsz
            lx = cx - tw // 2
            ly = cy - th // 2 + i * (th + 2)
            # thick black shadow for contrast
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
                drw.text((lx+dx, ly+dy), line, fill=(0, 0, 0), font=font)
            drw.text((lx, ly), line, fill=col, font=font)

    return np.array(img_pil)


# --- full pipeline ---
def run_analysis(filepath: str,
                 blue_ch_idx: int = 1,
                 red_ch_idx: int = 0,
                 amas_channel_indices: list = None,
                 amas_detection_mode: str = "auto",
                 amas_cellpose_diameter: int = None,
                 nucleus_diameter=None,
                 flow_threshold: float = 0.4,
                 cellprob_threshold: float = -3.5,
                 amas_min_size: int = 2,
                 amas_max_size: int = 150,
                 threshold_method: str = "otsu",
                 dilation_radius: int = 100,
                 min_amastigotes: int = 1,
                 avg_amastigote_size: int = 15,
                 hardware_mode: str = "auto",
                 amas_color: str = "red",
                 progress_callback=None) -> dict:
    
    """Run the full analysis on one image.

    amas_channel_indices: channels holding the parasite signal. None uses
        red_ch_idx; [0, 1] combines red and green with a pixel-wise max.
    amas_detection_mode: "auto" | "cellpose" | "classic"
        auto     - choose from the signal density (CellPose if sparse,
                   classic if dense)
        cellpose - always CellPose, classic as a fallback
        classic  - always the classic method
    amas_cellpose_diameter: amastigote diameter for CellPose (None = auto)
    """

    def progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    global USE_GPU
    USE_GPU = _check_gpu(hardware_mode)

    import time as _time
    _t_total = _time.time()

    progress("Loading image...", 5)
    img, meta = load_image(filepath)

    progress("Extracting channels...", 15)
    blue, red = extract_channels(img, blue_ch_idx, red_ch_idx, amas_channel_indices, meta)

    progress("Segmenting nuclei...", 30)
    import time as _time
    _t_seg = _time.time()

    # derive the cytoplasm radius when it was left at the default (100):
    # nucleus radius * 2.5 (cytoplasm reaches ~1.5x the nucleus radius)
    if dilation_radius == 100:  # default value = not set by the user
        auto_diam    = _estimate_nucleus_diameter(blue)
        auto_dil_rad = int((auto_diam / 2) * 2.5)
        auto_dil_rad = max(30, min(300, auto_dil_rad))
        print(f"[Cytoplasm radius] auto: nucleus diameter={auto_diam}px -> radius={auto_dil_rad}px")
        dilation_radius = auto_dil_rad

    nuc_mask = segment_nuclei(blue,
                              diameter=nucleus_diameter,
                              flow_threshold=flow_threshold,
                              cellprob_threshold=cellprob_threshold)
    print(f"[Time] nucleus segmentation: {_time.time() - _t_seg:.1f}s")

    progress("Detecting amastigotes...", 60)

    # choose the detection method
    used_method = "classic"
    result_cp   = None

    if amas_detection_mode in ("cellpose", "auto"):
        # auto mode: decide from the signal density
        if amas_detection_mode == "auto":
            use_cp = _should_use_cellpose_for_amastigotes(red)
        else:
            use_cp = True

        # On CPU, CellPose for amastigotes is very slow and the classic
        # method works better for small spots, so use classic unless the
        # user explicitly asks for CellPose on a GPU.
        if use_cp and not USE_GPU and amas_detection_mode == "auto":
            print("[Pipeline] CPU: amastigotes with the classic method")
            use_cp = False

        if use_cp:
            progress("Detecting amastigotes (CellPose)...", 62)
            result_cp = detect_amastigotes_cellpose(
                red,
                min_size=amas_min_size,
                max_size=amas_max_size,
                expected_diameter=amas_cellpose_diameter
            )
            if result_cp is not None:
                sparse_mask, dense_mask, dense_analysis = result_cp
                used_method = "cellpose"
                print(f"[Pipeline] detection: CellPose ({sparse_mask.max()} objects)")
            else:
                print("[Pipeline] CellPose found nothing, falling back to the classic method")

    if used_method == "classic":
        sparse_mask, dense_mask, dense_analysis = detect_amastigotes(
            red, amas_min_size, amas_max_size, threshold_method)
        print(f"[Pipeline] detection: classic ({sparse_mask.max()} objects)")

    dense_analysis['detection_method'] = used_method

    progress("Computing infection...", 80)
    results = compute_infection(
        nuc_mask, sparse_mask, dense_mask, dense_analysis,
        dilation_radius=dilation_radius,
        min_amastigotes=min_amastigotes,
        avg_amastigote_size=avg_amastigote_size)

    progress("Building overlay...", 90)
    overlay = build_overlay(blue, red, nuc_mask, sparse_mask, dense_mask, results, amas_color)

    progress("Analysis complete", 100)

    is_dense = any(c["mode"] in ["low_dense", "medium_dense", "high_dense"]
                   for c in results["per_cell"])

    print(f"[Time] total: {_time.time() - _t_total:.1f}s "
          f"({'GPU' if USE_GPU else 'CPU'})")

    return {
        "metadata":        meta,
        "blue_channel":    blue,
        "red_channel":     red,
        "nucleus_mask":    nuc_mask,
        "amastigote_mask": sparse_mask,
        "dense_mask":      dense_mask,
        "dense_analysis":  dense_analysis,
        "is_dense":        is_dense,
        "results":         results,
        "overlay":         overlay,
        "detection_method": used_method,
        "params": {
            "blue_ch_idx": blue_ch_idx, "red_ch_idx": red_ch_idx,
            "amas_channel_indices": amas_channel_indices,
            "amas_detection_mode": amas_detection_mode,
            "amas_cellpose_diameter": amas_cellpose_diameter,
            "nucleus_diameter": nucleus_diameter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "amas_min_size": amas_min_size, "amas_max_size": amas_max_size,
            "threshold_method": threshold_method,
            "dilation_radius": dilation_radius,
            "min_amastigotes": min_amastigotes,
            "avg_amastigote_size": avg_amastigote_size
        }
    }
