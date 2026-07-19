#!/usr/bin/env python3
"""
Extract fitness labels from the NatComm 2026 source data.

Reads Supp.Fig. 6 from source_data.xlsx and produces fitness_labels.csv
with one row per (variant, substrate) pair.

Source: Supp.Fig. 6 in the original paper's supplementary data.
The "Normalized fitness" values are the paper's published fitness scores,
already normalized relative to WT.

Usage:
    python3 scripts/build_fitness_labels.py
    python3 scripts/build_fitness_labels.py --output data/fitness_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "source_data.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "fitness_labels.csv"
SHEET_NAME = "Supp.Fig. 6"

# ── Column mapping: substrate → column name in Supp.Fig. 6 ──
FITNESS_COLUMNS = {
    "D-Ala": "Normalized fitness D-Ala",
    "D-Phe": "Normalized fitness D-Phe",
    "D-Met": "Normalized fitness D-Met",
    "D-Asn": "Normalized fitness D-Asn",
    "D-Gln": "Normalized fitness D-Gln",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract fitness labels from NatComm 2026 source data."
    )
    parser.add_argument("--source-xlsx", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser


def make_sample_id(variant_name: str, substrate_name: str) -> str:
    return f"{variant_name}__{substrate_name}"


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.source_xlsx, sheet_name=args.sheet_name)

    required_cols = ["Wild type residue ", "DAOx position", "Mutant residu"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {args.sheet_name}: {missing}")

    records: list[dict] = []
    for _, row in df.iterrows():
        wt_residue = str(row["Wild type residue "]).strip()
        position_value = row["DAOx position"]
        mutant_residue = str(row["Mutant residu"]).strip()

        # Skip invalid rows
        if (
            not wt_residue
            or wt_residue.lower() == "nan"
            or pd.isna(position_value)
            or not mutant_residue
            or mutant_residue.lower() == "nan"
        ):
            continue

        position = int(position_value)
        variant_name = f"{wt_residue}{position}{mutant_residue}"

        for substrate_name, fitness_column in FITNESS_COLUMNS.items():
            fitness_value = row.get(fitness_column)
            if pd.isna(fitness_value):
                continue
            records.append({
                "sample_id": make_sample_id(variant_name, substrate_name),
                "variant_name": variant_name,
                "substrate_name": substrate_name,
                "wt_residue": wt_residue,
                "position": position,
                "mutant_residue": mutant_residue,
                "fitness": float(fitness_value),
            })

    result = pd.DataFrame(records)
    result.to_csv(args.output_csv, index=False)
    print(f"fitness_labels.csv → {args.output_csv}")
    print(f"  {len(result)} rows ({len(result)//5} variants × 5 substrates)")
    print(f"  Fitness range: [{result['fitness'].min():.4f}, {result['fitness'].max():.4f}]")
    print(f"  Fitness mean:  {result['fitness'].mean():.4f}")


if __name__ == "__main__":
    main()
