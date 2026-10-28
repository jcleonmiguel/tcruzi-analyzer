"""
═══════════════════════════════════════════════════════════════════════════════
LIMPIADOR DE IMÁGENES DE MICROSCOPÍA — T. cruzi Analyzer
═══════════════════════════════════════════════════════════════════════════════

Limpia imágenes TIF/ND2 sin procesar para que el analyzer las detecte con
máxima precisión. Reproduce el efecto de tus imágenes editadas manualmente:
fondo negro puro + núcleos/amastigotes nítidos.

Pipeline:
  1. Background subtraction (elimina fondo difuso / autofluorescencia)
  2. Realce de contraste adaptativo
  3. Reducción de ruido suave
  4. Normalización al rango completo

USO:
  python limpiador_imagenes.py                    → procesa carpeta actual
  python limpiador_imagenes.py /ruta/a/imagenes   → procesa esa carpeta
  python limpiador_imagenes.py imagen.tif         → procesa una imagen

Las imágenes limpias se guardan en una subcarpeta "limpias/" con el sufijo "_clean".
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import tifffile
except ImportError:
    print("ERROR: falta tifffile. Instala con: pip install tifffile")
    sys.exit(1)

from skimage import filters, exposure


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN (ajustable)
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Background subtraction
    "bg_sigma_nuclei": 30,      # sigma gaussiana para fondo de núcleos (px)
    "bg_sigma_amas": 8,         # sigma para fondo de amastigotes (px)

    # Normalización
    "norm_low_pct": 40,         # percentil bajo (recorta fondo)
    "norm_high_pct": 99.8,      # percentil alto (preserva picos)

    # Realce de contraste local (CLAHE)
    "use_clahe": True,
    "clahe_clip": 0.02,

    # Reducción de ruido
    "denoise_sigma": 0.6,       # suavizado gaussiano leve para quitar ruido

    # Canales (para imágenes RGB/multicanal)
    # None = auto-detectar; o especificar índices
    "nuclei_channel": None,     # canal de núcleos (azul/DAPI)
    "amas_channel": None,       # canal de amastigotes
}


def _load_image(filepath):
    """Carga TIF o ND2 y devuelve array + nombres de canal."""
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
            print("  ERROR: falta nd2. Instala con: pip install nd2")
            return None, None
    elif ext in (".tif", ".tiff"):
        img = tifffile.imread(filepath)
    else:
        return None, None

    return img, channel_names


def _get_channels(img, channel_names):
    """
    Separa la imagen en canal de núcleos y canal de amastigotes.
    Devuelve (nuclei_raw, amas_raw, layout_info).
    """
    arr = img.astype(np.float32)
    if arr.ndim == 4:
        arr = arr.max(axis=0)

    # Determinar layout
    if arr.ndim == 2:
        # Imagen de un solo canal: usar el mismo para ambos (caso raro)
        return arr, arr, "single"

    # Multicanal: (C,H,W) o (H,W,C)
    if arr.shape[0] <= 8:        # (C, H, W)
        n_ch = arr.shape[0]
        get = lambda i: arr[i]
        channels_last = False
    else:                        # (H, W, C)
        n_ch = arr.shape[2]
        get = lambda i: arr[:, :, i]
        channels_last = True

    # --- Descartar canal ALFA en imágenes RGBA ---
    # Un alfa típico está saturado (casi todo 255) y sin estructura; engaña
    # a la heurística de "canal más brillante" robándole los núcleos.
    if n_ch == 4:
        alpha = get(3)
        if alpha.min() >= 250 or alpha.std() < 1.0:
            n_ch = 3

    # --- Caso RGB ya compuesto (TIF en color) ---
    # Convención del propio limpiador al guardar: R=amastigotes, B=núcleos.
    # Si es exactamente RGB de 3 canales y no hay nombres ND2, usar esa
    # convención directamente en lugar de adivinar por intensidad.
    if n_ch == 3 and channels_last and not channel_names:
        r = get(0); g = get(1); b = get(2)
        # núcleos = azul; amastigotes = rojo o verde, el que tenga señal real.
        #
        # IMPORTANTE: NO usar "pico/media" para decidir. Un canal casi vacío
        # (p.ej. el verde, solo ruido) tiene media ~0 y por tanto un pico/media
        # enorme, lo que hacía elegir por error el canal vacío y borrar todos
        # los amastigotes (caso C13/A15).
        #
        # Criterio robusto: CANTIDAD de señal fuerte real (nº de píxeles que
        # superan un umbral significativo). El canal con amastigotes reales
        # tiene decenas de miles de píxeles brillantes; un canal vacío tiene
        # casi ninguno. Se mide sobre varios umbrales para ser estable.
        def _signal_strength(ch):
            # píxeles claramente por encima del ruido de fondo típico
            return int((ch > 50).sum()) + int((ch > 80).sum())

        sr = _signal_strength(r)
        sg = _signal_strength(g)
        if max(sr, sg) < 20:
            # ningún canal tiene señal real de amastigotes → usar el más brillante
            amas, amas_name = (r, "ch0(R)") if r.mean() >= g.mean() else (g, "ch1(G)")
        elif sr >= sg:
            amas, amas_name = r, "ch0(R)"
        else:
            amas, amas_name = g, "ch1(G)"
        return b, amas, f"RGB compuesto: nuc=ch2(B), amas={amas_name}"

    # Auto-detección por nombre (ND2)
    nuc_idx, amas_idx = None, None
    if channel_names:
        for i, name in enumerate(channel_names):
            nl = name.lower()
            if any(x in nl for x in ['blue', 'dapi', 'hoechst', 'qblue', 'cy5']):
                nuc_idx = i
            elif any(x in nl for x in ['red', 'qred', 'green', 'gfp', 'fitc',
                                        'qgreen', 'cy3', 'tritc', 'alexa']):
                amas_idx = i

    # Si no se detectó por nombre, usar heurística por intensidad
    if nuc_idx is None or amas_idx is None:
        means = [float(get(i).mean()) for i in range(n_ch)]
        maxs  = [float(get(i).max()) for i in range(n_ch)]

        # Canal de núcleos = el más brillante en promedio (DAPI intenso y extenso)
        if nuc_idx is None:
            nuc_idx = int(np.argmax(means))

        # Canal de amastigotes = mayor señal puntual (max/mean alto) PERO
        # descartando canales casi vacíos (max < 40 = solo ruido)
        if amas_idx is None:
            candidates = [i for i in range(n_ch)
                          if i != nuc_idx and maxs[i] >= 40]
            if candidates:
                # De los canales con señal real, el de mayor pico relativo
                ratios = [(maxs[i] / (means[i] + 1e-6), i) for i in candidates]
                amas_idx = max(ratios)[1]
            else:
                # Ningún canal claro → usar el segundo más brillante
                order = sorted(range(n_ch), key=lambda i: means[i], reverse=True)
                amas_idx = order[1] if len(order) > 1 else nuc_idx

    return get(nuc_idx), get(amas_idx), f"nuc=ch{nuc_idx}, amas=ch{amas_idx}"


def _clean_nuclei(ch):
    """Limpia el canal de núcleos: quita fondo difuso, realza, normaliza."""
    ch = ch.astype(np.float32)

    # 1. Background subtraction (gaussiana grande)
    bg = filters.gaussian(ch, sigma=CONFIG["bg_sigma_nuclei"], preserve_range=True)
    corr = np.clip(ch - bg, 0, None)

    # 2. Reducción de ruido leve
    if CONFIG["denoise_sigma"] > 0:
        corr = filters.gaussian(corr, sigma=CONFIG["denoise_sigma"], preserve_range=True)

    # 3. Normalización por percentiles
    if corr.max() > 0:
        lo = float(np.percentile(corr, CONFIG["norm_low_pct"]))
        hi = float(np.percentile(corr, CONFIG["norm_high_pct"]))
        norm = np.clip((corr - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        norm = corr

    # 4. CLAHE para realce local de contraste
    if CONFIG["use_clahe"] and norm.max() > 0:
        norm = exposure.equalize_adapthist(norm, clip_limit=CONFIG["clahe_clip"])

    return (norm * 255).astype(np.uint8)


def _clean_amastigotes(ch):
    """Limpia el canal de amastigotes: realza puntos brillantes, elimina fondo."""
    ch = ch.astype(np.float32)

    # 1. Background subtraction fuerte (sigma pequeño para puntos)
    bg = filters.gaussian(ch, sigma=CONFIG["bg_sigma_amas"], preserve_range=True)
    corr = np.clip(ch - bg, 0, None)

    # 2. Umbral de ruido: eliminar fondo difuso, preservar solo picos reales
    cvals = corr[corr > 0]
    if len(cvals) > 10:
        p99 = float(np.percentile(corr, 99))
        noise_floor = max(15.0, p99 * 0.5)
        # Normalizar preservando solo señal por encima del ruido
        hi = max(float(np.percentile(corr, 99.9)), noise_floor + 10)
        norm = np.clip((corr - noise_floor) / (hi - noise_floor + 1e-6), 0, 1)
        norm_arr = (norm * 255)
        norm_arr[corr < noise_floor] = 0
        return norm_arr.astype(np.uint8)
    else:
        return np.zeros_like(ch, dtype=np.uint8)


def _detect_blue_amastigotes(blue_raw, red_raw, amas_clean):
    """
    Detecta amastigotes que se tiñen de AZUL (captan DAPI/Hoechst en su ADN)
    y que el canal rojo no recoge. Devuelve una máscara uint8 (0/valor) con
    SOLO los amastigotes azules PUROS, lista para fusionar al canal rojo.

    Es CONSERVADOR a propósito: exige puntos compactos, brillantes y del
    tamaño de un amastigote, para no confundir la textura de cromatina de los
    núcleos reales con amastigotes. Y descarta cualquier punto azul que ya
    tenga señal roja (amastigote) encima, para NO contar dos veces el mismo
    parásito (un amastigote rojo con algo de azul por debajo).

    Parámetros del filtro (moderadamente conservadores):
      - top-hat con disco de radio 7 (realza objetos pequeños, ignora núcleos
        grandes y difusos)
      - umbral percentil 99.0 (capta más amastigotes azules sin disparar ruido)
      - área 5–120 px², forma compacta (solidez > 0.80, excentricidad < 0.88)
      - intensidad azul media > 60
    """
    from skimage import measure, morphology
    from skimage.morphology import white_tophat, disk
    import numpy as _np

    blue = blue_raw.astype(_np.float32)
    red  = red_raw.astype(_np.float32)

    # Top-hat: resalta puntos pequeños brillantes sobre el fondo nuclear difuso.
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

    # Señal roja fuerte: zonas donde YA hay un amastigote rojo detectado.
    # Usamos tanto el rojo crudo como el ya limpiado para no duplicar.
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
        # ¿Tiene señal roja encima? → ya contado como rojo, NO duplicar.
        cy, cx = [int(v) for v in p.centroid]
        y0, y1 = max(0, cy - 3), cy + 4
        x0, x1 = max(0, cx - 3), cx + 4
        if red_strong[y0:y1, x0:x1].any():
            continue
        # Amastigote azul puro: lo marcamos con intensidad alta fija (como un
        # amastigote rojo nítido), para que el detector lo cuente igual.
        for yy, xx in p.coords:
            out[yy, xx] = 255
        n_pure += 1

    return out


def clean_image(filepath, output_dir):
    """Procesa una imagen y guarda la versión limpia."""
    name = Path(filepath).stem
    print(f"\n→ Procesando: {Path(filepath).name}")

    img, channel_names = _load_image(filepath)
    if img is None:
        print(f"  ✗ Formato no soportado")
        return False

    nuc_raw, amas_raw, layout = _get_channels(img, channel_names)
    print(f"  Canales: {layout}")
    if channel_names:
        print(f"  Nombres ND2: {channel_names}")

    # Limpiar cada canal
    nuc_clean  = _clean_nuclei(nuc_raw)
    amas_clean = _clean_amastigotes(amas_raw)

    print(f"  Núcleos limpios: max={nuc_clean.max()}, std={nuc_clean.std():.1f}")
    print(f"  Amastigotes limpios: pixels>0={100*(amas_clean>0).sum()/amas_clean.size:.2f}%")

    # Componer imagen RGB de salida:
    #   Canal 0 (R) = amastigotes
    #   Canal 2 (B) = núcleos
    h, w = nuc_clean.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = amas_clean   # rojo = amastigotes
    rgb[:, :, 2] = nuc_clean    # azul = núcleos

    # Guardar
    out_path = Path(output_dir) / f"{name}_clean.tif"
    tifffile.imwrite(str(out_path), rgb)
    print(f"  ✓ Guardada: {out_path.name}")
    return True


def clean_image_to_path(filepath, out_path, verbose=False):
    """
    Limpia UNA imagen y la guarda exactamente en `out_path` (sin forzar
    carpeta ni sufijo). Pensada para ser llamada desde la GUI.

    Estandariza SIEMPRE la salida a la convención del analyzer:
        canal R (0) = amastigotes (vengan en rojo o en verde en el original)
        canal B (2) = núcleos
    Así el análisis posterior puede detectar siempre en rojo.

    Devuelve un dict con métricas, o None si el formato no es soportado.
    """
    img, channel_names = _load_image(str(filepath))
    if img is None:
        return None

    nuc_raw, amas_raw, layout = _get_channels(img, channel_names)
    if verbose:
        print(f"  Canales: {layout}")

    nuc_clean  = _clean_nuclei(nuc_raw)
    amas_clean = _clean_amastigotes(amas_raw)

    # Detectar amastigotes AZULES puros (los que el canal rojo no recoge) y
    # fusionarlos al canal de amastigotes. Así se cuentan igual que los rojos,
    # sin duplicar los que ya tienen señal roja encima.
    blue_amas = _detect_blue_amastigotes(nuc_raw, amas_raw, amas_clean)
    n_blue = int((blue_amas > 0).sum())
    if n_blue > 0:
        amas_clean = np.maximum(amas_clean, blue_amas)
        # Quitarlos del canal de NÚCLEOS: si se dejan, su señal azul forma un
        # enjambre que CellPose agruparía como un núcleo gigante falso (el caso
        # del "núcleo 50"). Al borrarlos del azul, CellPose ya no los confunde
        # con material nuclear y quedan contados solo como amastigotes en rojo.
        nuc_clean = nuc_clean.copy()
        nuc_clean[blue_amas > 0] = 0
        if verbose:
            print(f"  Amastigotes azules añadidos: {n_blue} px (quitados del canal núcleos)")

    h, w = nuc_clean.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = amas_clean   # rojo = amastigotes (estandarizado)
    rgb[:, :, 2] = nuc_clean    # azul = núcleos

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
    # Determinar entrada
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "."

    target_path = Path(target)

    # Recopilar archivos a procesar
    if target_path.is_file():
        files = [target_path]
        base_dir = target_path.parent
    elif target_path.is_dir():
        files = sorted([f for f in target_path.iterdir()
                        if f.suffix.lower() in (".tif", ".tiff", ".nd2")])
        base_dir = target_path
    else:
        print(f"ERROR: '{target}' no existe")
        return

    if not files:
        print(f"No se encontraron imágenes TIF/ND2 en {target}")
        return

    # Crear carpeta de salida
    output_dir = base_dir / "limpias"
    output_dir.mkdir(exist_ok=True)

    print("═" * 70)
    print(f"LIMPIADOR DE IMÁGENES — {len(files)} archivo(s)")
    print(f"Salida: {output_dir}")
    print("═" * 70)

    ok = 0
    for f in files:
        try:
            if clean_image(f, output_dir):
                ok += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print("\n" + "═" * 70)
    print(f"COMPLETADO: {ok}/{len(files)} imágenes limpiadas")
    print(f"Guardadas en: {output_dir}")
    print("═" * 70)


if __name__ == "__main__":
    main()
