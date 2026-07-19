#!/usr/bin/env python3
"""
Compute distance from each DAOx residue CA to FAD N5 atom.

Input:  WT_model.pdb (PDB structure)
        daox_position_map.csv (paper position ↔ PDB residue number)
Output: data/distance.csv (position, residue, ca_to_fad_n5_A)

All distances are Euclidean distances in Ångströms, computed from the
WT crystal structure coordinates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDB = ROOT / "data" / "pdb" / "WT_model.pdb"
DEFAULT_MAP = ROOT / "data" / "pdb" / "daox_position_map.csv"
DEFAULT_OUT = ROOT / "data" / "distance.csv"


def parse_pdb_atoms(pdb_path: Path):
    """Parse ATOM/HETATM records from a PDB file.

    Returns list of (record_type, atom_name, residue_name, chain, resseq, x, y, z).
    """
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            record = line[0:6].strip()
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append((record, atom_name, residue_name, chain, resseq, x, y, z))
    return atoms


def find_fad_n5(atoms: list) -> np.ndarray:
    """Locate FAD N5 atom coordinates. FAD is HETATM with residue name FAD."""
    for rec, aname, rname, chain, resseq, x, y, z in atoms:
        if rname == "FAD" and aname == "N5":
            return np.array([x, y, z])
    # Fallback: try other names
    for rec, aname, rname, chain, resseq, x, y, z in atoms:
        if rname == "FAD" and "N5" in aname:
            return np.array([x, y, z])
    raise RuntimeError("FAD N5 atom not found in PDB file")


def main():
    parser = argparse.ArgumentParser(
        description="Compute residue-to-FAD distances."
    )
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--posmap", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # 1. Parse PDB
    atoms = parse_pdb_atoms(args.pdb)
    fad_n5 = find_fad_n5(atoms)
    print(f"FAD N5 coordinates: ({fad_n5[0]:.2f}, {fad_n5[1]:.2f}, {fad_n5[2]:.2f})")

    # 2. Read position map (paper position → PDB residue number)
    pos_map = pd.read_csv(args.posmap)
    paper_to_pdb = dict(zip(pos_map["source_position"], pos_map["pdb_residue_number"]))
    pdb_to_paper = {v: k for k, v in paper_to_pdb.items()}
    print(f"Position map: {len(paper_to_pdb)} entries")

    # 3. Compute CA-FAD distance for each residue
    rows = []
    for rec, aname, rname, chain, resseq, x, y, z in atoms:
        if aname != "CA":
            continue
        if rname == "FAD":
            continue  # skip FAD itself
        paper_pos = pdb_to_paper.get(resseq)
        if paper_pos is None:
            continue  # not in our position map
        coord = np.array([x, y, z])
        dist = float(np.linalg.norm(coord - fad_n5))
        rows.append({
            "position": paper_pos,
            "pdb_residue_number": resseq,
            "wt_residue": rname,
            "ca_to_fad_n5_A": round(dist, 2),
        })

    df = pd.DataFrame(rows).sort_values("position")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Output: {args.output} ({len(df)} positions, range {df['ca_to_fad_n5_A'].min():.1f}-{df['ca_to_fad_n5_A'].max():.1f}Å)")


if __name__ == "__main__":
    main()
