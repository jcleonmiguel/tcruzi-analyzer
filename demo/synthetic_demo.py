"""
Synthetic sanity check for TCruzi Analyzer.

Generates a two-channel fluorescence-like image with a KNOWN ground truth
(25 host cells, 10 of them infected with 3-8 amastigotes each), runs the full
analysis pipeline in CPU mode and compares the result with the ground truth.
No lab data and no GPU are needed.

Run from the repository root:
    python demo/synthetic_demo.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyzer  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
IMG_PATH = OUT_DIR / "synthetic.tif"


def make_synthetic_image(seed=0, size=1024):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    blue = np.zeros((size, size), np.float32)
    red = np.zeros((size, size), np.float32)
    centers = [(100 + 200 * i + rng.integers(-20, 20), 100 + 200 * j + rng.integers(-20, 20))
               for i in range(5) for j in range(5)]
    for cy, cx in centers:  # nuclei: elliptical blobs ~ 34 px diameter
        blue += 200 * np.exp(-(((yy - cy) / 18.0) ** 2 + ((xx - cx) / 15.0) ** 2) ** 2)
    truth = {}
    for k in rng.choice(len(centers), 10, replace=False):  # 10 infected cells
        cy, cx = centers[k]
        n = int(rng.integers(3, 9))
        truth[int(k)] = n
        for _ in range(n):  # amastigotes in the cytoplasm around the nucleus
            ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(28, 55)
            py, px = cy + r * np.sin(ang), cx + r * np.cos(ang)
            red += 220 * np.exp(-(((yy - py) ** 2 + (xx - px) ** 2) / (2 * 1.6 ** 2)))
    blue += rng.normal(8, 3, blue.shape)
    red += rng.normal(6, 3, red.shape)
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[..., 0] = np.clip(red, 0, 255)   # R = amastigotes
    rgb[..., 2] = np.clip(blue, 0, 255)  # B = nuclei
    return rgb, centers, truth


def main():
    rgb, centers, truth = make_synthetic_image()
    tifffile.imwrite(str(IMG_PATH), rgb)

    t0 = time.time()
    out = analyzer.run_analysis(str(IMG_PATH), blue_ch_idx=2, red_ch_idx=0,
                                amas_detection_mode="classic", hardware_mode="cpu",
                                dilation_radius=120, avg_amastigote_size=0)
    elapsed = time.time() - t0
    res = out["results"]
    cells = res["per_cell"]

    # Match each true cell to the closest detected nucleus and compare counts
    tree = cKDTree([(c["centroid_y"], c["centroid_x"]) for c in cells])
    exact = 0
    for k, (cy, cx) in enumerate(centers):
        _, j = tree.query((cy, cx))
        exact += int(cells[j]["amastigotes_count"] == truth.get(k, 0))

    detected_amas = sum(c["amastigotes_count"] for c in cells)
    print("\n" + "=" * 60)
    print(f"{'':<22}{'Ground truth':>14}{'Detected':>12}")
    print(f"{'Cells':<22}{len(centers):>14}{res['total_cells']:>12}")
    print(f"{'Infected cells':<22}{len(truth):>14}{res['infected_cells']:>12}")
    print(f"{'Infection (%)':<22}{100 * len(truth) / len(centers):>14.1f}{res['infection_pct']:>12.1f}")
    print(f"{'Amastigotes':<22}{sum(truth.values()):>14}{detected_amas:>12}")
    print(f"Cells with exact amastigote count: {exact}/{len(centers)}")
    print(f"Runtime (CPU): {elapsed:.1f} s")
    print("=" * 60)

    try:
        from PIL import Image
        Image.fromarray(out["overlay"]).save(OUT_DIR / "synthetic_overlay.png")
        print(f"Overlay saved to {OUT_DIR / 'synthetic_overlay.png'}")
    except Exception as e:
        print(f"Could not save overlay: {e}")


if __name__ == "__main__":
    main()
