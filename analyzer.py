"""
T. cruzi Amastigote Infection Analyzer — v7
Pipeline completo: CellPose GPU + detección híbrida disperso/denso por célula.
"""

import numpy as np
from pathlib import Path
import warnings


def _remove_small(binary, size):
    """
    Elimina objetos menores que 'size'. Compatible con todas las versiones de
    skimage (nuevas usan max_size, viejas min_size).
    """
    from skimage import morphology as sk_m
    try:
        return sk_m.remove_small_objects(binary, max_size=size)
    except TypeError:
        return sk_m.remove_small_objects(binary, min_size=size)
warnings.filterwarnings('ignore')

# ─── GPU check ────────────────────────────────────────────────────────────────
def _check_gpu(force_mode="auto") -> bool:
    '''
    Detecta GPU con soporte para modo forzado
    Args:
        force_mode: "auto", "gpu", o "cpu"
    '''
    if force_mode == "cpu":
        print("[Hardware] 🖥 Modo CPU FORZADO por usuario")
        return False
    elif force_mode == "gpu":
        # Verificar que la GPU es realmente utilizable (PyTorch con CUDA).
        # En el ejecutable empaquetado PyTorch puede no incluir CUDA: en ese caso
        # forzar GPU haría fallar CellPose, así que se vuelve a CPU con aviso.
        try:
            from cellpose import core
            if core.use_gpu():
                print("[Hardware] ⚡ Modo GPU FORZADO por usuario")
                return True
        except Exception:
            pass
        print("[Hardware] ⚠ GPU solicitada pero no disponible (PyTorch sin CUDA) "
              "→ se usa CPU")
        return False
    else:  # auto
        try:
            from cellpose import core
            gpu = core.use_gpu()
            if gpu:
                print("[Hardware] ✓ GPU detectada - Modo GPU activado")
            else:
                print("[Hardware] CPU detectado - Modo CPU activado")
            return gpu
        except Exception:
            print("[Hardware] CPU por defecto")
            return False

USE_GPU = _check_gpu()  # Default auto al inicio

# ─── Aceleración CPU: downscale para segmentación de núcleos ───────────────────
# En CPU, CellPose escala con el nº de píxeles. Los núcleos son grandes (~60px),
# así que segmentarlos a media resolución da resultados equivalentes en ~1/4 del
# tiempo. La máscara resultante se reescala al tamaño original con vecino más
# cercano (preserva las etiquetas enteras de cada núcleo).
# Solo se activa en CPU y solo si la imagen es grande y los núcleos no son diminutos.
CPU_DOWNSCALE_ENABLED  = True     # poner False para desactivar por completo
CPU_DOWNSCALE_MIN_DIM  = 1000     # solo si la dimensión menor supera esto
CPU_DOWNSCALE_TARGET   = 768      # dimensión menor objetivo tras reducir (cpsam CPU)
CPU_DOWNSCALE_MIN_DIAM = 24       # no reducir si los núcleos miden < 24px

# ─── Modo CPU rápido: segmentar núcleos sin CellPose ───────────────────────────
# cpsam (CellPose 4.x) en CPU tarda MINUTOS por imagen y es inviable para
# desplegar en equipos sin GPU. El watershed clásico segmenta núcleos en
# SEGUNDOS con calidad equivalente para núcleos redondeados típicos (DAPI).
# Cuando está activo y se corre en CPU, se usa watershed clásico directamente
# y se omite cpsam por completo. En GPU se usa siempre CellPose (rápido).
CPU_FAST_NUCLEI = True   # True = watershed clásico en CPU (rápido)
                          # False = CellPose también en CPU (lento, calidad cpsam)


def _compute_downscale_factor(shape, diameter, use_gpu):
    """
    Decide el factor de reducción para la segmentación de núcleos en CPU.
    Devuelve 1.0 (sin reducir) o un factor < 1.0.

    Nota: cpsam (CellPose 4.x) en CPU es muy pesado, por eso el objetivo es
    agresivo (768px). Los núcleos grandes (~50px) se segmentan bien incluso a
    esa resolución porque seguirán midiendo ~19px tras reducir.
    """
    if use_gpu or not CPU_DOWNSCALE_ENABLED:
        return 1.0
    min_dim = min(shape[0], shape[1])
    if min_dim <= CPU_DOWNSCALE_MIN_DIM:
        return 1.0
    # Si los núcleos ya son pequeños, reducir los haría indetectables
    if diameter is not None and diameter < CPU_DOWNSCALE_MIN_DIAM:
        return 1.0
    factor = CPU_DOWNSCALE_TARGET / min_dim
    factor = max(0.25, min(1.0, factor))  # no reducir más de 4x
    return factor


def _downscale_for_nuclei(blue_channel, factor):
    """Reduce el canal azul. Devuelve (imagen_reducida, shape_original)."""
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
    """Reescala una máscara etiquetada al tamaño original (vecino más cercano)."""
    if tuple(mask.shape) == tuple(orig_shape):
        return mask
    from skimage.transform import resize
    up = resize(mask, orig_shape, order=0, preserve_range=True,
                anti_aliasing=False).astype(np.int32)
    return up


# ─── CellPose model cache ──────────────────────────────────────────────────────
_CELLPOSE_MODEL = None
_CELLPOSE_MODEL_GPU = None
_CELLPOSE_IS_V4 = None  # se detecta una vez


def _cellpose_is_v4():
    """Detecta si CellPose es 4.x+ (API CellposeModel/cpsam) o 2.x (API Cellpose)."""
    global _CELLPOSE_IS_V4
    if _CELLPOSE_IS_V4 is None:
        from cellpose import models
        # En 4.x no existe models.Cellpose; en 2.x sí
        _CELLPOSE_IS_V4 = not hasattr(models, "Cellpose")
        try:
            import cellpose
            print(f"[CellPose] Versión {cellpose.version} "
                  f"({'API 4.x/cpsam' if _CELLPOSE_IS_V4 else 'API 2.x'})")
        except Exception:
            pass
    return _CELLPOSE_IS_V4


def _eval_cellpose(model, image, diameter, flow_threshold, cellprob_threshold):
    """
    Llama a model.eval de forma compatible con CellPose 2.x y 4.x.
    Devuelve siempre solo 'masks'.
    - 2.x: eval(..., channels=[0,0]) y devuelve 4 valores
    - 4.x (cpsam): sin 'channels', devuelve 3 valores, imagen monocanal directa
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
    return out[0]  # masks es siempre el primer elemento


def _get_cellpose_model():
    global _CELLPOSE_MODEL, _CELLPOSE_MODEL_GPU
    if _CELLPOSE_MODEL is None or _CELLPOSE_MODEL_GPU != USE_GPU:
        from cellpose import models
        print(f"[CellPose] Cargando modelo de núcleos (GPU: {USE_GPU})...")
        if _cellpose_is_v4():
            # CellPose 4.x: modelo único cpsam (cell + nucleus, generalista)
            _CELLPOSE_MODEL = models.CellposeModel(gpu=USE_GPU)
            _CELLPOSE_MODEL_GPU = USE_GPU
            print("[CellPose] ✓ Modelo cpsam (4.x) cargado correctamente")
        else:
            # CellPose 2.x: modelo 'nuclei', fallback a 'cyto2'
            try:
                _CELLPOSE_MODEL = models.Cellpose(gpu=USE_GPU, model_type='nuclei')
                _CELLPOSE_MODEL_GPU = USE_GPU
                print("[CellPose] ✓ Modelo 'nuclei' (2.x) cargado correctamente")
            except Exception as e:
                print(f"[CellPose] ⚠ Error cargando 'nuclei': {e}, probando 'cyto2'...")
                try:
                    _CELLPOSE_MODEL = models.Cellpose(gpu=USE_GPU, model_type='cyto2')
                    _CELLPOSE_MODEL_GPU = USE_GPU
                    print("[CellPose] ✓ Modelo 'cyto2' (2.x) cargado correctamente")
                except Exception as e2:
                    raise RuntimeError(
                        f"No se pudo cargar ningún modelo de CellPose: {e2}\n"
                        "Verifica que los modelos estén descargados correctamente."
                    )
    return _CELLPOSE_MODEL


# ─── Image I/O ────────────────────────────────────────────────────────────────
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
    """Estandarización universal: BG subtraction + normalización adaptativa."""
    from skimage import exposure as sk_exp, filters as sk_f

    ch = ch.astype(np.float32)
    bit_max   = 65535.0 if ch.max() > 255 else 255.0
    sig_ratio = float(ch.max()) / bit_max
    frac_nz   = float((ch > 0).sum()) / ch.size

    print(f"  [{mode}] max={ch.max():.0f}/{bit_max:.0f} ({100*sig_ratio:.1f}%), p50={np.percentile(ch,50):.0f}")

    if ch.max() == 0:
        return np.zeros_like(ch, dtype=np.uint8)

    # BG subtraction siempre para núcleos (elimina fondo difuso/gradientes)
    if mode == "nuclei":
        bg_s  = max(10, min(ch.shape) // 20)
        corr  = np.clip(ch - sk_f.gaussian(ch, sigma=bg_s), 0, None)
        # CLAHE SIEMPRE: ecualización local de contraste. Recupera núcleos
        # tenues/oscuros que un simple stretch p1-p99 aplastaría hasta hacerlos
        # invisibles para CellPose. clip_limit bajo evita amplificar ruido.
        corr_01 = corr / (corr.max() + 1e-6)
        corr    = sk_exp.equalize_adapthist(corr_01, clip_limit=0.01) * 255
        # Stretch suave de respaldo (percentiles amplios para no recortar señal)
        lo, hi = np.percentile(corr, 0.5), np.percentile(corr, 99.8)
        if hi <= lo: hi = float(corr.max())
        result = np.clip((corr - lo) / (hi - lo + 1e-6) * 255, 0, 255)
        print(f"  [{mode}] BG_sub(sigma={bg_s}) + CLAHE → std={result.std():.1f}")

    else:  # amastigotes
        # Los amastigotes son PUNTOS BRILLANTES. El fondo difuso NO cuenta.
        # Background subtraction + umbral absoluto: los amastigotes reales superan
        # ~20 tras quitar fondo; el fondo difuso queda por debajo de ~10.
        bg_s = max(8, min(ch.shape) // 40)
        corr = np.clip(ch - sk_f.gaussian(ch, sigma=bg_s), 0, None)

        cvals = corr[corr > 0]
        if len(cvals) < 10:
            return np.zeros_like(ch, dtype=np.uint8)

        # Umbral de detección: combina absoluto mínimo con percentil alto
        # para adaptarse a distinto brillo pero sin capturar fondo difuso
        p99   = float(np.percentile(corr, 99))
        p999  = float(np.percentile(corr, 99.9))
        # El umbral debe estar bien por encima del fondo:
        # mínimo absoluto 15, o el p99 si la imagen tiene más señal
        noise_floor = max(15.0, p99 * 0.6)

        hi = max(p999, noise_floor + 10)
        # Mapear: noise_floor → 30, hi → 255; por debajo del floor → 0
        result = np.clip((corr - noise_floor) / (hi - noise_floor + 1e-6) * 225 + 30, 0, 255)
        result[corr < noise_floor] = 0
        print(f"  [{mode}] BG_sub + umbral_abs: noise_floor={noise_floor:.1f}, hi={hi:.1f}")

    out = result.astype(np.uint8)
    print(f"  [{mode}] output: mean={out.mean():.1f}, std={out.std():.1f}, "
          f"pixels>0={100*(out>0).sum()/out.size:.3f}%")
    return out


def _estimate_nucleus_diameter(blue_channel: np.ndarray) -> int:
    """
    Mide el diámetro real de los núcleos en la imagen.
    Funciona con cualquier zoom de microscopio.
    """
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
    print(f"[CellPose] Diámetro medido en imagen: {estimated}px "
          f"(n={len(valid)}, rango {min(diameters):.0f}-{max(diameters):.0f}px)")
    return estimated


def extract_channels(img: np.ndarray, blue_ch: int, red_ch: int,
                     amas_channel_indices: list = None,
                     meta: dict = None):
    """Extrae y estandariza canales. Funciona con cualquier imagen."""
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

    # Auto-detección ND2
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
                print(f"[Canales] ND2 auto: núcleos={ab}, amas={ar}")
                blue_ch, red_ch = ab, ar
                amas_channel_indices = [ar]
        except Exception as e:
            print(f"[Canales] ND2 auto falló: {e}")

    blue_raw = _get_ch(blue_ch)
    if amas_channel_indices and len(amas_channel_indices) > 0:
        channels = [_get_ch(i).astype(np.float32) for i in amas_channel_indices]
        red_raw  = np.max(np.stack(channels, axis=0), axis=0)
    else:
        red_raw = _get_ch(red_ch)

    print(f"[Canales] Núcleos=canal{blue_ch}, Amastigotes=canal{red_ch}")
    return _standardize_channel(blue_raw, "nuclei"), _standardize_channel(red_raw, "amastigotes")


# ─── Nucleus Segmentation ─────────────────────────────────────────────────────
def _count_nuclei_classic(blue_channel: np.ndarray, diameter: int) -> int:
    """
    Estima el número de núcleos por detección clásica (watershed).
    Sirve como referencia para validar la detección de CellPose.
    """
    from skimage import filters as sk_f, morphology as sk_m, measure as sk_meas, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return 0

    sm  = sk_f.gaussian(ch, sigma=4)
    thr = sk_f.threshold_otsu(sm)
    binary = sm > (thr * 0.6)  # umbral permisivo para incluir tenues
    min_area = int(np.pi * (diameter/2)**2 * 0.3)  # 30% del área esperada
    binary = _remove_small(binary, max(50, min_area))

    # Watershed para separar núcleos pegados
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
    """
    Segmentación de núcleos por watershed clásico.
    Robusto para núcleos tenues que CellPose no detecta.
    Detecta núcleos como objetos brillantes sobre fondo oscuro.
    """
    from skimage import filters as sk_f, morphology as sk_m, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return np.zeros_like(ch, dtype=np.int32)

    if diameter is None:
        diameter = _estimate_nucleus_diameter(ch)

    # Suavizado y umbral permisivo (captura núcleos tenues)
    sm  = sk_f.gaussian(ch, sigma=4)
    thr = sk_f.threshold_otsu(sm)
    binary = sm > (thr * 0.55)

    # Limpiar: quitar objetos pequeños (amastigotes), rellenar huecos
    min_area = int(np.pi * (diameter/2)**2 * 0.25)
    binary = _remove_small(binary, max(100, min_area))
    binary = sk_m.closing(binary, sk_m.disk(3))
    binary = ndimage.binary_fill_holes(binary)

    # Watershed para separar núcleos pegados
    dist = ndimage.distance_transform_edt(binary)
    # Suavizar la transformada de distancia (proporcional al diámetro): elimina
    # los picos espurios que crean los nucleolos/textura interna y que harían
    # partir un núcleo en dos. Sin esto, núcleos con nucleolo se sobre-cuentan.
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

    print(f"[Watershed] {ws.max()} núcleos detectados (diam={diameter}px, min_dist={min_dist})")
    return ws.astype(np.int32)


def _split_merged_nuclei(masks: np.ndarray, diameter: int) -> np.ndarray:
    """
    Separa núcleos que CellPose fusionó en una sola máscara.
    Detecta etiquetas anormalmente grandes (cúmulos pegados) y las parte por
    watershed sobre la transformada de distancia. No toca las máscaras normales.
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
        # Solo intervenir si la máscara es claramente mayor que 1 núcleo
        if r.area < expected_area * 1.6:
            continue
        sub = (masks[r.slice] == r.label)
        dist = ndi.distance_transform_edt(sub)
        # Suavizado proporcional al diámetro: ignora nucleolos/textura interna
        # para no partir un núcleo en dos por sus huecos internos.
        dist = sk_f.gaussian(dist, sigma=max(1.0, diameter * 0.08))
        min_d = max(4, int(diameter * 0.30))
        coords = peak_local_max(dist, min_distance=min_d, labels=sub,
                                exclude_border=False)
        if len(coords) < 2:
            continue  # un solo núcleo, no partir
        seeds = np.zeros(dist.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndi.label(seeds)
        ws = segmentation.watershed(-dist, markers, mask=sub)
        n_parts = int(ws.max())
        if n_parts < 2:
            continue
        # Reescribir: primer fragmento conserva la etiqueta, el resto nuevas
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
        print(f"[Split] {split_count} núcleos fusionados separados → total {out.max()}")
    return out


def _rescue_missed_nuclei(masks: np.ndarray, blue_channel: np.ndarray,
                          diameter: int) -> np.ndarray:
    """
    Recupera núcleos reales que CellPose descartó por tener forma difusa o
    irregular (no por falta de brillo). Busca regiones de señal azul que NO
    quedaron etiquetadas, las valida por tamaño y las añade como núcleos nuevos.
    Núcleos grandes se parten por watershed para no subcontar.
    """
    from skimage import filters as sk_f, measure as sk_meas, morphology as sk_m, segmentation
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    import math

    ch = blue_channel.astype(np.float32)
    if ch.max() == 0:
        return masks

    nuc_area = math.pi * (diameter / 2) ** 2

    # Máscara permisiva de TODA la señal de núcleo (capta también los difusos)
    thr = sk_f.threshold_otsu(ch)
    signal = ch > thr * 0.55
    signal = ndi.binary_fill_holes(signal)
    signal = _remove_small(signal, int(nuc_area * 0.4))

    # Restar lo que CellPose ya detectó (con margen para no re-capturar bordes)
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
        # Si la región es grande, partir en varios núcleos plausibles
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
        print(f"[Rescate] {rescued} núcleos difusos recuperados → total {out.max()}")
    return out


def segment_nuclei(blue_channel: np.ndarray,
                   diameter=None,
                   flow_threshold: float = 0.4,
                   cellprob_threshold: float = 0.0) -> np.ndarray:
    """
    Segmenta núcleos con CellPose.
    
    IMPORTANTE: En CellPose 2.2.3, diameter=None (auto) a veces falla.
    Si diameter=None, estimamos un valor razonable basado en la imagen.
    """
    
    if diameter is None:
        diameter = _estimate_nucleus_diameter(blue_channel)
        print(f"[CellPose] diameter=auto → {diameter}px (medido en imagen)")
    
    # Verificar que la imagen tenga contenido
    if blue_channel.max() == 0:
        print("[CellPose] ⚠ WARNING: Canal azul completamente negro")
        return np.zeros_like(blue_channel, dtype=np.int32)
    
    if blue_channel.std() < 5:
        print(f"[CellPose] ⚠ WARNING: Canal azul con muy poco contraste (std={blue_channel.std():.1f})")

    # ── MODO CPU RÁPIDO: watershed clásico en lugar de cpsam ────────────────────
    # cpsam en CPU tarda minutos. El watershed clásico da núcleos en segundos.
    # Solo aplica en CPU; en GPU se usa CellPose (más abajo).
    if CPU_FAST_NUCLEI and not USE_GPU:
        print(f"[CPU-Rápido] Segmentando núcleos por watershed clásico "
              f"(diámetro={diameter}px) — evitando cpsam en CPU")
        try:
            masks = segment_nuclei_watershed(blue_channel, diameter)
            num_nuclei = int(masks.max())
            # Separar cúmulos y rescatar núcleos difusos (igual que la ruta CellPose)
            try:
                masks = _split_merged_nuclei(masks, diameter)
                num_nuclei = int(masks.max())
            except Exception as e:
                print(f"[Split] Falló: {e}, manteniendo máscaras sin separar")
            try:
                masks = _rescue_missed_nuclei(masks, blue_channel, diameter)
                num_nuclei = int(masks.max())
            except Exception as e:
                print(f"[Rescate] Falló: {e}, manteniendo máscaras sin rescate")
            print(f"[CPU-Rápido] ✓ Segmentación final: {num_nuclei} núcleos detectados")
            return masks.astype(np.int32)
        except Exception as e:
            print(f"[CPU-Rápido] ⚠ Watershed falló ({e}), usando CellPose como respaldo...")
            # Si falla, continúa con la ruta CellPose normal de abajo

    print(f"[CellPose] Parámetros: diameter={diameter}, flow_thr={flow_threshold}, cellprob_thr={cellprob_threshold}")
    
    model = _get_cellpose_model()

    # ── Aceleración CPU: reducir resolución para la segmentación de núcleos ──
    # CellPose corre sobre la versión reducida; el diámetro se escala igual.
    # En GPU o imágenes pequeñas, factor=1.0 (sin cambios).
    ds_factor = _compute_downscale_factor(blue_channel.shape, diameter, USE_GPU)
    if ds_factor < 0.999:
        seg_input, _orig_shape = _downscale_for_nuclei(blue_channel, ds_factor)
        # diámetro escalado, con suelo de 16px (cpsam funciona bien desde ~15px)
        seg_diameter = max(16, int(round(diameter * ds_factor)))
        print(f"[Acelera-CPU] Núcleos a {ds_factor:.2f}x: "
              f"{blue_channel.shape}→{seg_input.shape}, diámetro {diameter}→{seg_diameter}px")
    else:
        seg_input, _orig_shape = blue_channel, blue_channel.shape
        seg_diameter = diameter

    try:
        # Primer intento con los parámetros dados (compatible 2.x/4.x)
        masks = _eval_cellpose(model, seg_input, seg_diameter,
                               flow_threshold, cellprob_threshold)
        num_nuclei = int(masks.max())
        print(f"[CellPose] Intento 1 (cellprob={cellprob_threshold}, flow={flow_threshold}): {num_nuclei} núcleos")

        # Estimar cuántos núcleos esperamos por detección clásica
        # (la estimación clásica usa la imagen reducida también, para coherencia)
        expected = _count_nuclei_classic_safe(seg_input, seg_diameter)
        print(f"[CellPose] Núcleos esperados (estimación clásica): ~{expected}")

        # RE-INTENTO ÚNICO ADAPTATIVO: si CellPose detecta menos de lo esperado,
        # un solo reintento con cellprob bajo recupera núcleos tenues.
        # (Antes probaba -1,-2,-3 = hasta 3 pasadas extra. Una sola pasada bien
        #  elegida captura casi lo mismo a una fracción del tiempo en CPU.)
        if expected > 0 and num_nuclei < expected * 0.90:
            retry_thr = min(-2.0, cellprob_threshold - 1.0)
            print(f"[CellPose] Detección baja ({num_nuclei}<{int(expected*0.90)}), "
                  f"reintento único cellprob={retry_thr}...")
            try:
                m2 = _eval_cellpose(model, seg_input, seg_diameter,
                                    flow_threshold, retry_thr)
                n2 = int(m2.max())
                print(f"[CellPose]   → {n2} núcleos")
                if n2 > num_nuclei and n2 <= max(expected * 1.5, num_nuclei + 1):
                    masks, num_nuclei = m2, n2
            except Exception as e:
                print(f"[CellPose]   reintento falló: {e}")

        # Reescalar la máscara al tamaño original ANTES de watershed/split/rescate
        # (esos pasos trabajan sobre el canal azul en resolución completa)
        if ds_factor < 0.999:
            masks = _upscale_mask(masks, blue_channel.shape)
            num_nuclei = int(masks.max())
            print(f"[Acelera-CPU] Máscara reescalada a {blue_channel.shape}: {num_nuclei} núcleos")

        # Fallback watershed solo si seguimos MUY por debajo
        # (se recalcula expected en resolución completa para comparar bien)
        expected_full = _count_nuclei_classic_safe(blue_channel, diameter)
        if expected_full > 0 and num_nuclei < expected_full * 0.80:
            print(f"[CellPose] Aún baja ({num_nuclei}), probando WATERSHED...")
            try:
                ws_masks = segment_nuclei_watershed(blue_channel, diameter)
                num_ws = int(ws_masks.max())
                print(f"[Watershed] {num_ws} núcleos vs CellPose {num_nuclei}")
                if num_ws > num_nuclei:
                    masks = ws_masks
                    num_nuclei = num_ws
                    print(f"[CellPose] ✓ Usando WATERSHED (mejor detección de núcleos tenues)")
            except Exception as e:
                print(f"[Watershed] Falló: {e}, manteniendo CellPose")

        # SEPARAR CÚMULOS: parte núcleos pegados que quedaron como una sola máscara
        try:
            masks = _split_merged_nuclei(masks, diameter)
            num_nuclei = int(masks.max())
        except Exception as e:
            print(f"[Split] Falló: {e}, manteniendo máscaras sin separar")

        # RESCATE: recuperar núcleos difusos/irregulares que CellPose descartó
        # por forma (no por brillo). cellprob más bajo no los recupera.
        try:
            masks = _rescue_missed_nuclei(masks, blue_channel, diameter)
            num_nuclei = int(masks.max())
        except Exception as e:
            print(f"[Rescate] Falló: {e}, manteniendo máscaras sin rescate")

        print(f"[CellPose] ✓ Segmentación final: {num_nuclei} núcleos detectados")
        
        if num_nuclei == 0:
            print("[CellPose] ⚠ WARNING: NO se detectaron núcleos")
            print("           Sugerencias:")
            print("           - Prueba cellprob_threshold más bajo (-1.0, -2.0)")
            print("           - Ajusta diameter (40-100 según tamaño de núcleos)")
        
        return masks.astype(np.int32)
        
    except Exception as e:
        print(f"[CellPose] ✗ ERROR durante segmentación: {e}")
        raise


# ─── CellPose Amastigote Detection ────────────────────────────────────────────
_CELLPOSE_CYTO_MODEL = None
_CELLPOSE_CYTO_GPU   = None

def _get_cellpose_cyto_model():
    """Carga (y cachea) el modelo cyto para detectar amastigotes."""
    global _CELLPOSE_CYTO_MODEL, _CELLPOSE_CYTO_GPU
    if _CELLPOSE_CYTO_MODEL is None or _CELLPOSE_CYTO_GPU != USE_GPU:
        from cellpose import models
        print(f"[CellPose-Amas] Cargando modelo cyto (GPU: {USE_GPU})...")
        if _cellpose_is_v4():
            # 4.x: un único modelo cpsam sirve para todo (núcleos y amastigotes)
            _CELLPOSE_CYTO_MODEL = models.CellposeModel(gpu=USE_GPU)
            _CELLPOSE_CYTO_GPU   = USE_GPU
            print("[CellPose-Amas] ✓ Modelo cpsam (4.x) cargado")
        else:
            for mtype in ("cyto3", "cyto2", "cyto"):
                try:
                    _CELLPOSE_CYTO_MODEL = models.Cellpose(gpu=USE_GPU, model_type=mtype)
                    _CELLPOSE_CYTO_GPU   = USE_GPU
                    print(f"[CellPose-Amas] ✓ Modelo '{mtype}' cargado")
                    break
                except Exception as e:
                    print(f"[CellPose-Amas] ⚠ '{mtype}' no disponible: {e}")
        if _CELLPOSE_CYTO_MODEL is None:
            raise RuntimeError("No se pudo cargar ningún modelo cyto para amastigotes")
    return _CELLPOSE_CYTO_MODEL


def detect_amastigotes_cellpose(amas_channel: np.ndarray,
                                 min_size: int = 2,
                                 max_size: int = 150,
                                 expected_diameter: int = None) -> tuple:
    """
    Detecta amastigotes usando CellPose (modelo cyto).

    Ideal cuando los amastigotes están separados y tienen morfología de célula
    pequeña. Filtra por forma, no solo por brillo → menos falsos positivos.

    Devuelve (sparse_mask, dense_mask, dense_analysis) igual que detect_amastigotes,
    o None si CellPose no detectó nada (el caller hace fallback al método clásico).
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
        print("[CellPose-Amas] Canal vacío — skipping")
        return None

    try:
        model  = _get_cellpose_cyto_model()
        masks  = _eval_cellpose(model, amas_channel, expected_diameter,
                                flow_threshold=0.6, cellprob_threshold=-2.0)
    except Exception as e:
        print(f"[CellPose-Amas] Error: {e}")
        return None

    n_detected = int(masks.max())
    print(f"[CellPose-Amas] Detectados: {n_detected} objetos")
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
    print(f"[CellPose-Amas] Filtrado: {n_sparse} sparse, {len(dense_regions_list)} densos")

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
    """
    Heurística para el modo 'auto':
    - Señal densa (>12% del área) → método clásico (más robusto con masas)
    - Señal dispersa              → CellPose (más preciso con objetos separados)
    """
    from skimage import filters
    arr = amas_channel.astype(np.float32)
    if arr.max() == 0 or arr.std() < 1:
        return False
    smoothed = filters.gaussian(arr, sigma=1.0)
    thr = float(np.percentile(smoothed, 95))
    signal_fraction = float((smoothed > thr).sum()) / float(smoothed.size)
    if signal_fraction > 0.12:
        print(f"[Auto-modo] Señal densa ({signal_fraction*100:.1f}%) → CLÁSICO")
        return False
    print(f"[Auto-modo] Señal dispersa ({signal_fraction*100:.1f}%) → CELLPOSE")
    return True


# ─── Amastigote Detection ──────────────────────────────────────────────────────
def detect_amastigotes(red_channel: np.ndarray,
                       min_size: int = 2,
                       max_size: int = 150,
                       threshold_method: str = "otsu"):
    """
    v10: detección sensible + separación simple discreto/denso.

    Idea clave:
    - Mantener la sensibilidad de la versión nueva para no perder puntos rojos.
    - No contar por área salvo que el componente sea una masa densa real.
    - Los puntos sueltos/pequeños clusters pasan a `sparse_mask` y se contarán
      después como objetos discretos o por picos locales capados.
    """
    from skimage import filters, morphology, measure

    arr = red_channel.astype(np.float32)
    mean_val = float(arr.mean())
    std_val  = float(arr.std())
    max_val  = float(arr.max())

    print(f"[Detección v12] Canal rojo — mean={mean_val:.1f}, std={std_val:.1f}, max={max_val:.1f}")

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

    # Realce rápido de puntos/masas sobre fondo: conserva más amastigotes que el
    # umbral clásico, pero no fusiona mediante cierres morfológicos agresivos.
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

    # Sensible, pero con piso para no incluir ruido/halo débil.
    # v13: piso y factor rebajados para capturar amastigotes de brillo MEDIO
    # (los que tienen algo de fluorescencia azul mezclada y por tanto su señal
    # roja es más tenue). Antes floor=p95*0.50 y otsu*0.78 anclaban el umbral a
    # los amastigotes más brillantes y dejaban fuera ~15% de brillo medio.
    # El filtro remove_small_objects de más abajo sigue eliminando el ruido,
    # así que bajar el umbral es seguro.
    floor = max(p95 * 0.35, float(smoothed.max()) * 0.035, 1.0)
    thresh_final = max(method_thr * 0.65, floor)
    thresh_final = min(thresh_final, max(p99 * 0.92, floor))

    print(f"[Detección v12] Candidatos — stat={stat:.1f}, otsu={thresh_otsu:.1f}, triangle={thresh_tri:.1f}, yen={thresh_yen:.1f}, p95={p95:.1f}, p99={p99:.1f}")
    print(f"[Detección v12] Umbral FINAL: {thresh_final:.1f}")

    binary = smoothed > thresh_final
    binary = morphology.remove_small_objects(binary, min_size=max(1, int(min_size)))

    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels, intensity_image=smoothed)
    if not regions:
        return _empty_analysis(thresh_final)

    # Calibración para distinguir tamaño unitario, NO para contar puntos sueltos.
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
            # Aceptamos tamaños más pequeños que en v9 para no perder detección,
            # pero no permitimos que un valor diminuto infle masas densas.
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

    # Separación simple: solo los componentes claramente grandes van a dense_mask.
    # El umbral es más alto que v8 para que 24 hpi con puntos sueltos no active masas.
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
        print(f"[Calibración v12] AUTO aceptada: {avg_area:.1f}px², n={n_filtered}, cv={cv if cv is not None else 0:.2f}, confianza={confidence_level}")
    else:
        print(f"[Calibración v12] AUTO no usada para masas (n={n_total}). Conteo discreto protegido.")

    print(f"[Detección v12] Sparse labels={int(sparse_mask.max())}, candidatos densos={len(dense_regions_list)}, dense_area≥{candidate_dense_area}px², masa real≥{true_dense_min_area}px²")

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


"""
Función adicional para separar clusters en calibración
Insertar después de detect_amastigotes
"""


# ─── Watershed para separar clusters ───────────────────────────────────────────
def separate_amastigote_clusters(sparse_mask: np.ndarray, 
                                 red_channel: np.ndarray,
                                 expected_size: float = 13) -> np.ndarray:
    """
    Separa clusters claros de amastigotes pegados usando watershed.

    v7: más conservador para no dividir ruido pequeño y para acelerar.
    Solo intenta separar objetos claramente mayores que un amastigote individual.
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

            # Primero intentamos máximos en señal; si no hay suficientes, usamos distancia.
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


# ─── Infection Criterion ──────────────────────────────────────────────────────
def compute_infection(nucleus_mask: np.ndarray,
                      sparse_mask: np.ndarray,
                      dense_mask: np.ndarray,
                      dense_analysis: dict,
                      dilation_radius: int = 100,
                      min_amastigotes: int = 1,
                      avg_amastigote_size: int = 15) -> dict:
    """
    v12: detección/asignación nuevas + conteo fino conservador.

    - Asigna cada señal roja al núcleo más cercano con KDTree.
    - Puntos sueltos: 1 componente = 1 amastigote.
    - Clusters no densos: se estiman por máximos locales, capados para no saltar
      artificialmente a >50.
    - Masa densa real: solo si el componente es muy grande y ocupa una fracción
      relevante del territorio celular; entonces se estima por área calibrada.
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

    # ── Calibración efectiva: solo para clusters/masas, nunca para puntos sueltos ──
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

    # Protección anti-sobreconteo: para estimar masas, nunca dividir por valores minúsculos.
    sparse_unit = max(float(effective_amas_size), 8.0)
    dense_unit = max(float(effective_amas_size) * 1.45, 16.0)
    candidate_dense_area = int(dense_analysis.get('candidate_dense_area') or max(600, sparse_unit * 38))
    true_dense_min_area = int(dense_analysis.get('true_dense_min_area') or max(1600, sparse_unit * 110))

    print(f"[Conteo v12] Calibración {calibration_source}: unit={effective_amas_size:.1f}px²; sparse_unit={sparse_unit:.1f}; dense_unit={dense_unit:.1f}")

    # Territorio aproximado por célula para validar masas densas sin distance_transform global.
    if len(nuc_props) > 1:
        nn_dists, _ = tree.query(nuc_centroid_arr, k=2)
        nearest_neighbor = nn_dists[:, 1]
    else:
        nearest_neighbor = np.array([2 * dilation_radius], dtype=float)

    territory_area = np.zeros(n_nuclei + 1, dtype=float)
    # Radio máximo de asignación por núcleo: borde del núcleo + margen de citosol.
    # Un amastigote más lejos que esto es "libre" y NO se asigna.
    max_assign_radius = np.zeros(n_nuclei + 1, dtype=float)
    for i, lbl in enumerate(nuc_labels):
        local_r = min(float(dilation_radius), max(18.0, float(nearest_neighbor[i]) * 0.55))
        territory_area[int(lbl)] = float(nuc_areas.get(int(lbl), 0)) + np.pi * local_r * local_r
        # Radio del núcleo a partir de su área real
        nuc_area = float(nuc_areas.get(int(lbl), 200))
        nuc_radius = np.sqrt(nuc_area / np.pi)
        # Margen de citosol: hasta dónde se asignan amastigotes alrededor del núcleo.
        # En T. cruzi los amastigotes llenan el citoplasma formando un anillo que se
        # extiende ~1.5x el radio del núcleo. Antes el margen era nearest_neighbor*0.6,
        # demasiado corto: dejaba fuera el anillo de células claramente infectadas
        # (p.ej. la 49) y las marcaba como sanas.
        #
        # Nueva lógica:
        #   - suelo biológico: 1.5 x radio del núcleo (cubre el citoplasma típico)
        #   - techo por vecino: nearest_neighbor * 0.75 (no invadir células contiguas)
        #   - el margen final respeta el suelo aunque el vecino esté cerca, pero
        #     nunca supera dilation_radius.
        cito_floor = nuc_radius * 1.5
        nbr_limit  = float(nearest_neighbor[i]) * 0.75
        cito_margin = max(cito_floor, nbr_limit)
        cito_margin = min(cito_margin, float(dilation_radius))
        max_assign_radius[int(lbl)] = nuc_radius + cito_margin

    def _owner_from_region(reg) -> int:
        cy, cx = reg.centroid
        dist, idx = tree.query([float(cy), float(cx)], k=1)
        owner = int(nuc_labels[int(idx)])
        # Solo asignar si el amastigote está dentro del radio de esa célula.
        # Si está más lejos → amastigote libre en el fondo, NO se asigna.
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
        """Detecta máximos brillantes, pero con parámetros conservadores.

        En v10 los máximos locales podían multiplicarse dentro de un mismo
        amastigote al usar min_distance=3 y percentiles bajos. Esta versión se
        usa solo como respaldo: el conteo principal viene de núcleos brillantes
        separados por umbral local alto.
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

        # Umbral alto para no sacar varios picos dentro de un mismo punto.
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
        """Cuenta amastigotes visibles como centros brillantes separados.

        Ajuste v12: para señal discreta NO se usa
        área / tamaño medio, porque eso infla mucho los valores cuando el halo o
        el suavizado hacen que 2 puntos formen un componente grande. En su lugar
        se subumbraliza cada componente con un umbral local alto; así dos puntos
        pegados siguen siendo dos, no 12-14.
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
        # v12: umbral local más alto. En v11 un único punto con halo podía
        # partirse en dos cores y duplicar el conteo. Aquí nos quedamos con el
        # centro realmente brillante y exigimos cores algo más sólidos.
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

        # Filtrar cores diminutos: un solo amastigote puede tener 1-2 píxeles
        # máximos separados por ruido. Solo cuentan como centros independientes
        # los cores con área suficiente o brillo claramente dominante.
        valid = []
        for cp in core_props:
            imax = float(getattr(cp, 'intensity_max', 0.0))
            if cp.area >= 3 or (cp.area >= 2 and imax >= vmax * 0.88):
                valid.append(cp)

        return int(len(valid))

    def _count_discrete_component(label_img: np.ndarray, reg) -> int:
        """Cuenta puntos/clusters no densos con un modelo realista.

        Regla v12:
        - Punto pequeño: 1.
        - Cluster discreto: número de centros brillantes separados.
        - Nunca dividir área entre tamaño de amastigote para señal discreta.
          Esa división era la causa de contar 2 puntos como 14.
        - Solo las masas densas verdaderas, validadas más abajo, se estiman por
          área y pueden entrar en 15-50 o >50.
        """
        area = float(reg.area)
        # v12: elevar el umbral de "un solo amastigote". En tus imágenes de
        # 24hpi los puntos discretos tienen halo; en v11 ese halo hacía que un
        # punto aislado pasara a contar 2. Ahora, todo componente pequeño cuenta
        # como 1 salvo que sea claramente un cluster grande.
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
        # El respaldo por picos ahora es más estricto: solo se usa si el objeto
        # es lo bastante grande/alargado para representar varios amastigotes.
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

        # Cap geométrico estricto. Para rango 1-15 es mejor quedarse corto por
        # 1 amastigote que convertir 2 puntos visibles en 4. Los clusters reales
        # grandes siguen pudiendo subir a 15-50 por número de centros separados.
        max_by_length = max(1, int(np.ceil(major_axis / diameter_guess)))
        max_by_area = max(1, int(np.ceil(area / max(28.0, sparse_unit * 3.2))))
        if minor_axis > diameter_guess * 1.7 and area > min_single_area * 2.5:
            # cluster compacto bidimensional: permitir algo más que la longitud.
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

    # ── 1) Discretos: asignación nueva, conteo protegido ───────────────────────
    amas_props = measure.regionprops(sparse_mask, intensity_image=det_img)
    assigned_sparse = 0
    # Para la 2ª pasada (contigüidad): mapa label_amastigote → célula dueña,
    # y lista de amastigotes que quedaron libres en la 1ª pasada.
    owner_of_amas = {}          # reg.label → owner_id (solo asignados)
    free_regions = []           # regiones sin dueño tras la 1ª pasada
    for reg in amas_props:
        owner_id = _owner_from_region(reg)
        if owner_id == 0:
            free_regions.append(reg)   # candidato para la 2ª pasada
            continue  # amastigote libre en el fondo, de momento no se asigna
        owner_of_amas[reg.label] = owner_id
        n_obj = _count_discrete_component(sparse_mask, reg)
        if n_obj <= 0:
            continue
        counts_sparse[owner_id] += n_obj
        counts_total[owner_id] += n_obj
        assigned_sparse += n_obj

    # ── 1b) Segunda pasada por CONTIGÜIDAD ──────────────────────────────────────
    # Un amastigote libre que está físicamente pegado a un cúmulo ya asignado a
    # la célula X pertenece a esa misma infección (los amastigotes de una célula
    # forman un cúmulo conectado). Esto recupera los amastigotes del borde externo
    # del anillo sin asignar amastigotes realmente aislados en el fondo.
    # Es robusto en todas las imágenes porque se basa en contigüidad real, no en
    # un radio fijo que habría que calibrar por foto.
    if free_regions and owner_of_amas:
        from skimage import morphology as sk_morph
        from scipy import ndimage as sk_ndi

        # Mapa de qué célula posee cada píxel de amastigote asignado.
        owner_pixels = np.zeros_like(sparse_mask, dtype=np.int32)
        for lbl_a, own in owner_of_amas.items():
            owner_pixels[sparse_mask == lbl_a] = own

        # Distancia de contacto: proporcional al tamaño del amastigote unitario,
        # de modo que se autoajusta a la escala de cada imagen. El factor 3.0 une
        # amastigotes pegados al cúmulo (borde del anillo, separación ~1-1.5 diám.)
        # pero NO puentea amastigotes realmente aislados en el fondo (separación
        # grande), evitando asignaciones falsas. Calibrado sobre anillos típicos.
        link_dist = max(3, int(round(np.sqrt(max(sparse_unit, 4.0) / np.pi) * 3.0)))
        struct = sk_morph.disk(link_dist)

        # Resolvemos en varias iteraciones: un amastigote recién unido puede a su
        # vez "puentear" hacia otro libre más externo (cadena del anillo).
        changed = True
        passes = 0
        while changed and passes < 4 and free_regions:
            changed = False
            passes += 1
            still_free = []
            # Dilatamos el mapa de dueños para detectar contacto.
            owner_dil = sk_ndi.grey_dilation(owner_pixels, footprint=struct)
            for reg in free_regions:
                coords = reg.coords
                # Etiquetas de célula tocadas alrededor de este amastigote libre.
                touched = owner_dil[coords[:, 0], coords[:, 1]]
                touched = touched[touched > 0]
                if touched.size > 0:
                    # Dueño = célula mayoritaria en el contacto.
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

    # ── 1c) Tercera pasada: VECINO MÁS CERCANO con tope de distancia ───────────
    # Los amastigotes que siguen libres tras contigüidad suelen estar en el
    # espacio intercelular, claramente más cerca de UNA célula pero un poco
    # fuera de su radio estricto. Se asignan al núcleo más cercano siempre que
    # estén dentro de una distancia ampliada razonable (1.6x el radio de
    # asignación de ese núcleo). Amastigotes verdaderamente perdidos en el fondo
    # (más lejos que eso) siguen sin asignarse, evitando falsos positivos.
    if free_regions:
        EXTEND = 1.6   # cuánto se amplía el radio estricto en esta última pasada
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

    # ── 2) Candidatos densos: solo masas enormes pasan a conteo por área ───────
    dense_labels = measure.label(dense_mask, connectivity=2)
    dense_props = measure.regionprops(dense_labels, intensity_image=det_img)
    true_dense_components = 0
    fallback_dense_as_cluster = 0
    assigned_dense = 0

    for reg in dense_props:
        owner_id = _owner_from_region(reg)
        if owner_id == 0:
            continue  # masa densa libre en el fondo, no se asigna
        bbox = reg.bbox
        sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
        local_mask = dense_labels[sl] == reg.label
        signal_px = int((binary_signal[sl] & local_mask).sum())
        signal_area = max(signal_px, int(reg.area))
        terr = max(1.0, float(territory_area[owner_id] if territory_area[owner_id] > 0 else np.pi * dilation_radius ** 2))
        dense_fraction = signal_px / terr
        dense_fraction_by_cell[owner_id] = max(dense_fraction_by_cell[owner_id], float(dense_fraction))

        peaks = _local_peaks(dense_labels, reg, min_distance=3, percentile=62, max_peaks=650)

        # Masa densa verdadera: tamaño claramente grande + fracción local relevante.
        true_mass = (
            signal_area >= true_dense_min_area and
            (dense_fraction >= 0.08 or signal_area >= true_dense_min_area * 1.8) and
            (peaks >= 18 or signal_area >= true_dense_min_area * 2.2)
        )

        if true_mass:
            area_est = int(round(signal_area / dense_unit))
            # En masas reales permitimos >50, con límite biológico operativo.
            n_obj = max(15, min(max(area_est, peaks), 500))
            counts_dense[owner_id] += n_obj
            true_dense_components += 1
        else:
            # No era una masa real: tratar como cluster discreto capado.
            n_obj = _count_discrete_component(dense_labels, reg)
            counts_cluster[owner_id] += n_obj
            fallback_dense_as_cluster += 1

        counts_total[owner_id] += n_obj
        assigned_dense += n_obj

    print(f"[DEBUG v12] Componentes sparse detectados: {len(amas_props)}; conteo asignado: {assigned_sparse}")
    print(f"[DEBUG v12] Candidatos densos: {len(dense_props)}; masas reales: {true_dense_components}; clusters capados: {fallback_dense_as_cluster}; conteo denso/cluster: {assigned_dense}")
    print(f"[DEBUG v12] Total amastigotes reportado: {int(counts_total.sum())}")

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
            # >50 solo si viene de una masa densa real o de >50 componentes reales.
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


# ─── Overlay ──────────────────────────────────────────────────────────────────
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
    # Canal de amastigotes: pintar en rojo o verde según amas_color
    amas_signal = np.clip(red_ch.astype(np.uint16) * red_mult, 0, 255).astype(np.uint8)
    if amas_color == "green":
        rgb[:, :, 1] = amas_signal   # señal en canal verde
    else:
        rgb[:, :, 0] = amas_signal   # señal en canal rojo

    infected_ids = {c["nucleus_id"] for c in results["per_cell"] if c["infected"]}
    healthy_ids  = {c["nucleus_id"] for c in results["per_cell"] if not c["infected"]}

    # Contorno naranja de masas densas
    if dense_mask.any():
        db = bdil(segmentation.find_boundaries(dense_mask, mode='outer'),
                  structure=morphology.disk(max(2, min(h,w)//800)))
        rgb[db, 0] = 255; rgb[db, 1] = 140; rgb[db, 2] = 0

    # Relleno de núcleos: amarillo infectado, verde sano
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

    # Contorno blanco de todos los núcleos
    thick = max(2, min(h, w) // 600)
    tb = bdil(segmentation.find_boundaries(nucleus_mask, mode='outer'),
              structure=morphology.disk(thick))
    rgb[tb] = [255, 255, 255]

    # Contorno de amastigotes individuales (color según modo)
    if sparse_mask.max() > 0:
        ab = bdil(segmentation.find_boundaries(sparse_mask > 0, mode='outer'),
                  structure=morphology.disk(thick))
        if amas_color == "green":
            rgb[ab] = [0, 255, 0]      # Verde brillante
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
            # Sombra negra gruesa para máximo contraste
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
                drw.text((lx+dx, ly+dy), line, fill=(0, 0, 0), font=font)
            drw.text((lx, ly), line, fill=col, font=font)

    return np.array(img_pil)


# ─── Full Pipeline ─────────────────────────────────────────────────────────────
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
    
    """
    Pipeline completo de análisis.

    Nuevos parámetros:
    - amas_channel_indices: lista de índices de canales que contienen amastigotes.
        None → usa red_ch_idx (comportamiento legacy).
        [0, 1] → combina canales 0 y 1 (e.g. rojo + verde) con max píxel a píxel.
    - amas_detection_mode: "auto" | "cellpose" | "classic"
        auto     → decide según densidad de señal (CellPose si dispersa, clásico si densa)
        cellpose → fuerza CellPose para todos los casos; fallback a clásico si falla
        classic  → siempre usa el método clásico (comportamiento anterior)
    - amas_cellpose_diameter: diámetro en px para CellPose de amastigotes (None = auto)
    """

    def progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    global USE_GPU
    USE_GPU = _check_gpu(hardware_mode)

    import time as _time
    _t_total = _time.time()

    progress("Cargando imagen...", 5)
    img, meta = load_image(filepath)

    progress("Extrayendo canales...", 15)
    blue, red = extract_channels(img, blue_ch_idx, red_ch_idx, amas_channel_indices, meta)

    progress("Segmentando núcleos (CellPose)...", 30)
    import time as _time
    _t_seg = _time.time()

    # Auto-calcular radio citosol si no se especificó (=100 por defecto)
    # Radio citosol = radio_nucleo * 2.5 (citosol ocupa ~1.5x el radio del núcleo)
    if dilation_radius == 100:  # valor por defecto = no especificado manualmente
        auto_diam    = _estimate_nucleus_diameter(blue)
        auto_dil_rad = int((auto_diam / 2) * 2.5)
        auto_dil_rad = max(30, min(300, auto_dil_rad))
        print(f"[Radio citosol] Auto: diám_núcleo={auto_diam}px → radio_citosol={auto_dil_rad}px")
        dilation_radius = auto_dil_rad

    nuc_mask = segment_nuclei(blue,
                              diameter=nucleus_diameter,
                              flow_threshold=flow_threshold,
                              cellprob_threshold=cellprob_threshold)
    print(f"[Tiempo] Segmentación de núcleos: {_time.time() - _t_seg:.1f}s")

    progress("Detectando amastigotes...", 60)

    # ── Elegir método de detección ─────────────────────────────────────────────
    used_method = "classic"
    result_cp   = None

    if amas_detection_mode in ("cellpose", "auto"):
        # En modo auto, decidir por heurística de densidad
        if amas_detection_mode == "auto":
            use_cp = _should_use_cellpose_for_amastigotes(red)
        else:
            use_cp = True

        # En CPU, CellPose para amastigotes es muy lento y el método clásico
        # funciona mejor para puntos pequeños. Forzar clásico salvo que el
        # usuario lo pida explícitamente (modo "cellpose") teniendo GPU.
        if use_cp and not USE_GPU and amas_detection_mode == "auto":
            print("[Pipeline] CPU detectado → amastigotes por método clásico (más rápido y preciso)")
            use_cp = False

        if use_cp:
            progress("Detectando amastigotes (CellPose)...", 62)
            result_cp = detect_amastigotes_cellpose(
                red,
                min_size=amas_min_size,
                max_size=amas_max_size,
                expected_diameter=amas_cellpose_diameter
            )
            if result_cp is not None:
                sparse_mask, dense_mask, dense_analysis = result_cp
                used_method = "cellpose"
                print(f"[Pipeline] Detección: CellPose ✓ ({sparse_mask.max()} objetos)")
            else:
                print(f"[Pipeline] CellPose sin resultados → fallback a método clásico")

    if used_method == "classic":
        sparse_mask, dense_mask, dense_analysis = detect_amastigotes(
            red, amas_min_size, amas_max_size, threshold_method)
        print(f"[Pipeline] Detección: Clásico ({sparse_mask.max()} objetos)")

    dense_analysis['detection_method'] = used_method

    progress("Calculando infección...", 80)
    results = compute_infection(
        nuc_mask, sparse_mask, dense_mask, dense_analysis,
        dilation_radius=dilation_radius,
        min_amastigotes=min_amastigotes,
        avg_amastigote_size=avg_amastigote_size)

    progress("Construyendo overlay...", 90)
    overlay = build_overlay(blue, red, nuc_mask, sparse_mask, dense_mask, results, amas_color)

    progress("¡Análisis completo!", 100)

    is_dense = any(c["mode"] in ["low_dense", "medium_dense", "high_dense"]
                   for c in results["per_cell"])

    print(f"[Tiempo] TOTAL análisis imagen: {_time.time() - _t_total:.1f}s "
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
