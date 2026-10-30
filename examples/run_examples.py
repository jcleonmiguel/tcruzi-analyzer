"""Run the full analysis on the two example images, as the GUI does by
default (cleaning on, classic amastigote detection).

Writes an overlay per image and a results table to examples/output/.

    python examples/run_examples.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyzer import run_analysis          # noqa: E402
from image_cleaner import clean_image_to_path  # noqa: E402

OUT = HERE / "output"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for img in sorted(HERE.glob("*.tif")):
        with tempfile.TemporaryDirectory() as tmp:
            # cleaned images always have parasites in red (0), nuclei in blue (2)
            clean = Path(tmp) / f"{img.stem}_clean.tif"
            clean_image_to_path(img, clean)
            out = run_analysis(str(clean), blue_ch_idx=2, red_ch_idx=0,
                               amas_detection_mode="classic",
                               hardware_mode="auto", dilation_radius=120,
                               avg_amastigote_size=0)
        r = out["results"]
        Image.fromarray(out["overlay"]).save(OUT / f"{img.stem}_overlay.png")
        rows.append({
            "image": img.name,
            "cells": r["total_cells"],
            "infected": r["infected_cells"],
            "infected_pct": round(r["infection_pct"], 1),
            "amastigotes": sum(c["amastigotes_count"] for c in r["per_cell"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nOverlays and results.csv in {OUT}")


if __name__ == "__main__":
    main()
