#!/usr/bin/env python3
"""
Master data build script — generates all reproducible data files from source.

This script produces:
  1. fitness_labels.csv       — from source_data.xlsx Supp.Fig. 6
  2. substrate_encodings.csv  — from SMILES via RDKit (144 dims, deterministic)

These two files are the only data inputs needed by the LambdaRank pipeline
(besides the pre-computed ESM3 embeddings in pair_dataset.csv).

What CANNOT be generated here (requires external models):
  - ESM3 embeddings (1536d) — requires biohub/esm3-sm-open-v1 model inference
  - Docking features (~400d) — requires Rosetta RDock
  - pair_dataset.csv — is a JOIN of the above; symlinked from the main project

Usage:
    python3 scripts/build_all_data.py
    python3 scripts/build_all_data.py --data-dir data/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run_script(name: str) -> None:
    script = SCRIPTS / name
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR: {script.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    print("Building all reproducible data files...")
    print(f"Source: {ROOT / 'data' / 'source_data.xlsx'}")
    print()

    # Extract fitness labels from source Excel
    run_script("build_fitness_labels.py")

    # Generate substrate encodings from SMILES
    run_script("build_substrate_encodings.py")

    # print(f"\n{'='*60}")
    # print("All reproducible data files built successfully.")
    # print(f"{'='*60}")
    # print()
    # print("Files generated:")
    # print(f"  {ROOT / 'data' / 'fitness_labels.csv'}")
    # print(f"  {ROOT / 'data' / 'substrate_encodings.csv'}")
    # print()
    # print("NOT generated (requires external models):")
    # print("  - ESM3 embeddings → pair_dataset.csv (symlinked)")
    # print("  - Docking features → pair_dataset.csv (symlinked)")
    # print()
    # print("To verify the pipeline works end-to-end:")
    # print("  python3 lambdarank_ensemble.py")


if __name__ == "__main__":
    main()
