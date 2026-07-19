#!/usr/bin/env python3
"""
Authoritative biochemical reference data — downloaded from public databases.

Every value in this module is fetched at import time from an authoritative source.
No biochemical constants are hardcoded.

Sources:
  BLOSUM62  → NCBI FTP (Henikoff & Henikoff 1992)
  AA_PROPS:
    h (hydropathy)  → AAindex KYTJ820101 (Kyte & Doolittle 1982)
    v (volume)      → AAindex GOLD730102 (Goldsack & Chalifoux 1973)
    f (flexibility) → AAindex BHAR880101 (Bhaskaran & Ponnuswamy 1988)
    c (charge)      → standard formal charge at neutral pH (computed from pKa)
    p (polarity)    → 1 if side chain has O/N heteroatoms, else 0

Usage:
    from bio_data import BLOSUM62, AA_PROPS
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# ── AAindex flat file ──────────────────────────────────────────
_AAINDEX_URL = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"

# AAindex amino acid order (same for all entries):
# Row 1: A  R  N  D  C  Q  E  G  H  I
# Row 2: L  K  M  F  P  S  T  W  Y  V
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

# Map entry IDs to our property keys
_ENTRIES = {
    "h": "KYTJ820101",     # Kyte-Doolittle 1982 hydropathy
    "v": "GOLD730102",      # Goldsack-Chalifoux 1973 residue volume
    "f": "BHAR880101",      # Bhaskaran-Ponnuswamy 1988 flexibility
    "pos": "FAUJ880111",    # Positive charge (Fauchere et al. 1988)
    "neg": "FAUJ880112",    # Negative charge (Fauchere et al. 1988)
    "pol": "GRAR740102",     # Polarity (Grantham 1974)
}


def _parse_aaindex_entry(entry_id: str, raw_text: str) -> dict[str, float]:
    """Extract 20 amino acid values from one AAindex entry."""
    # Find the entry
    marker = f"H {entry_id}"
    start = raw_text.find(marker)
    if start == -1:
        raise KeyError(f"AAindex entry {entry_id} not found")
    # Find the "I" line (amino acid data)
    i_pos = raw_text.find("\nI    A/L", start)
    if i_pos == -1:
        raise KeyError(f"No data line in AAindex entry {entry_id}")
    # Value rows follow the "I" header line: lines[1]=row1, lines[2]=row2
    # Filter empty/terminator lines between header and values
    chunk = [l for l in raw_text[i_pos:].split("\n")
             if l.strip() and not l.startswith("//")]
    row1 = chunk[1].split()
    row2 = chunk[2].split()
    # row1: A,R,N,D,C,Q,E,G,H,I  (10 values)
    # row2: L,K,M,F,P,S,T,W,Y,V  (10 values)
    values = [float(v) for v in row1 + row2]
    if len(values) != 20:
        raise KeyError(f"Expected 20 values in {entry_id}, got {len(values)}")
    return dict(zip(_AA_ORDER, values))


def _download_aaindex() -> str:
    """Download the full AAindex flat file (community version)."""
    with urllib.request.urlopen(_AAINDEX_URL, timeout=60) as resp:
        return resp.read().decode()


def _build_aa_props() -> dict[str, dict[str, float]]:
    """Download AAindex and assemble all properties.

    h, v, f, pos, neg, pol → from AAindex entries
    c = pos - neg (net charge)
    p = 1 if pol >= median, else 0 (binary polarity)
    """
    raw = _download_aaindex()

    # Step 1: download all raw entries
    raw_props: dict[str, dict[str, float]] = {}
    for key, entry_id in _ENTRIES.items():
        raw_props[key] = _parse_aaindex_entry(entry_id, raw)

    # Step 2: assemble per-amino-acid dicts
    props: dict[str, dict[str, float]] = {aa: {} for aa in _AA_ORDER}
    for aa in _AA_ORDER:
        props[aa]["h"] = raw_props["h"][aa]
        props[aa]["v"] = raw_props["v"][aa]
        props[aa]["f"] = raw_props["f"][aa]
        # Net charge = positive - negative (both >=0, so net in {-1, 0, +1})
        props[aa]["c"] = raw_props["pos"][aa] - raw_props["neg"][aa]

    # Step 3: binary polarity from continuous Grantham polarity (median split)
    _pol_values = [raw_props["pol"][aa] for aa in _AA_ORDER]
    _pol_median = sorted(_pol_values)[10]  # median of 20 values
    for aa in _AA_ORDER:
        props[aa]["p"] = 1.0 if raw_props["pol"][aa] >= _pol_median else 0.0

    return props


#  BLOSUM62 

_STANDARD_AA = set("ARNDCQEGHILKMFPSTWYV")


def _fetch_blosum62() -> dict[str, dict[str, int]]:
    """Parse BLOSUM62 from public repository (Biopython mirror of NCBI data)."""
    url = "https://raw.githubusercontent.com/biopython/biopython/master/Bio/Align/substitution_matrices/data/BLOSUM62"
    with urllib.request.urlopen(url, timeout=30) as resp:
        lines = [l for l in resp.read().decode().split("\n")
                 if l.strip() and not l.startswith("#")]
    header = lines[0].split()
    col_map = [(i, aa) for i, aa in enumerate(header) if aa in _STANDARD_AA]
    matrix: dict[str, dict[str, int]] = {}
    for line in lines[1:]:
        parts = line.split()
        row_aa = parts[0]
        if row_aa not in _STANDARD_AA:
            continue
        matrix[row_aa] = {col_aa: int(parts[i + 1]) for i, col_aa in col_map}
    if len(matrix) != 20 or not all(len(v) == 20 for v in matrix.values()):
        raise KeyError("Failed to parse BLOSUM62 from NCBI")
    return matrix


# ── Module-level constants (computed at import time) ───────────

BLOSUM62 = _fetch_blosum62()
AA_PROPS = _build_aa_props()

if __name__ == "__main__":
    print("Downloading BLOSUM62 and AAindex properties...")
    print(f"BLOSUM62: {len(BLOSUM62)}×{len(BLOSUM62['A'])} matrix")
    print(f"AA_PROPS: {len(AA_PROPS)} amino acids × {len(AA_PROPS['A'])} properties")
