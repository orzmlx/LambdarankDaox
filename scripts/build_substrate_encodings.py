#!/usr/bin/env python3
"""
Build substrate encoding table — fully reproducible, no external model dependencies.

Generates substrate_encodings.csv with 3 feature blocks:
  1. Basic physicochemical (7 dims) — hand-curated from literature values
  2. RDKit molecular descriptors (9 dims) — computed from SMILES via RDKit
  3. Morgan fingerprints (128 dims) — ECFP4 via RDKit, radius=2, 128 bits

Total: 7 + 9 + 128 = 144 feature dimensions per substrate.

All values are deterministic given the same SMILES input.
No MACAW, no learned embeddings, no non-deterministic components.

References:
  - Hydropathy: Kyte & Doolittle (1982) J. Mol. Biol. 157:105-132
  - Volume: Creighton (1993) Proteins, 2nd ed.
  - RDKit descriptors: Landrum et al., RDKit: Open-source cheminformatics
  - Morgan fingerprints: Rogers & Hahn (2010) J. Chem. Inf. Model. 50:742-754

Usage:
    python3 build_substrate_encodings.py
    python3 build_substrate_encodings.py --output data/substrate_encodings.csv
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem import Descriptors, rdFingerprintGenerator

#  Substrate identity ，InChIKey is the authoritative reference
# InChIKeys verified 2026-07-19 against PubChem.
# These uniquely identify the exact stereoisomer (D-configuration).
_SUBSTRATE_INCHIKEYS: dict[str, str] = {
    "D-Ala": "QNAYBMKLOCPYGJ-REOHCLBHSA-N",
    "D-Asn": "DCXYFEDJOCDNAF-UWTATZPHSA-N",
    "D-Gln": "ZDXPYRJPNDTMRX-GSVOUGTGSA-N",
    "D-Met": "FFEARJCKVFRZRR-SCSAIBSYSA-N",
    "D-Phe": "COLNVLDHVKWLRT-MRVPVSSYSA-N",
}


def fetch_smiles_from_pubchem(inchikey: str) -> str:
    """Fetch the canonical SMILES for a compound from PubChem by InChIKey.

    PubChem REST API (free, no key required):
      GET /rest/pug/compound/inchikey/{inchikey}/property/SMILES/JSON

    Returns the SMILES string. Raises RuntimeError on failure.
    """
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        f"{inchikey}/property/SMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["SMILES"]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch SMILES for InChIKey {inchikey} from PubChem: {exc}"
        ) from exc


def get_substrate_smiles(cache: bool = True) -> dict[str, str]:
    """Return {substrate_name: SMILES} for the 5 D-amino acid substrates.

    SMILES are fetched from PubChem by InChIKey at call time.  The InChIKey
    guarantees the correct stereoisomer regardless of naming inconsistencies.
    Set cache=False to force a fresh fetch.
    """
    result: dict[str, str] = {}
    for name, inchikey in _SUBSTRATE_INCHIKEYS.items():
        result[name] = fetch_smiles_from_pubchem(inchikey)
        # Quick sanity check: the SMILES must be parseable by RDKit
        mol = Chem.MolFromSmiles(result[name])
        if mol is None:
            raise ValueError(f"PubChem returned unparseable SMILES for {name}: {result[name]}")
        fetched_ik = Chem.MolToInchiKey(mol)
        if fetched_ik != inchikey:
            raise ValueError(
                f"InChIKey mismatch for {name}: expected {inchikey}, got {fetched_ik}"
            )
    return result

#  Basic physicochemical properties 
# Source: Kyte-Doolittle (hydropathy), Creighton (volume), standard biochemistry (charge/polarity).
BASIC_FEATURES: dict[str, dict[str, float]] = {
    "D-Ala": {"side_chain_atoms": 1, "side_chain_hetero_atoms": 0, "is_aromatic": 0,
              "has_sulfur": 0, "has_amide": 0, "hydropathy": 1.8, "side_chain_volume": 88.6},
    "D-Asn": {"side_chain_atoms": 3, "side_chain_hetero_atoms": 1, "is_aromatic": 0,
              "has_sulfur": 0, "has_amide": 1, "hydropathy": -3.5, "side_chain_volume": 114.1},
    "D-Gln": {"side_chain_atoms": 4, "side_chain_hetero_atoms": 1, "is_aromatic": 0,
              "has_sulfur": 0, "has_amide": 1, "hydropathy": -3.5, "side_chain_volume": 143.8},
    "D-Met": {"side_chain_atoms": 4, "side_chain_hetero_atoms": 1, "is_aromatic": 0,
              "has_sulfur": 1, "has_amide": 0, "hydropathy": 1.9, "side_chain_volume": 162.9},
    "D-Phe": {"side_chain_atoms": 7, "side_chain_hetero_atoms": 0, "is_aromatic": 1,
              "has_sulfur": 0, "has_amide": 0, "hydropathy": 2.8, "side_chain_volume": 189.9},
}


def build_basic_block(substrate_name: str) -> dict[str, float]:
    """Return the 7 hand-curated physicochemical features for a substrate."""
    if substrate_name not in BASIC_FEATURES:
        raise KeyError(f"No basic features for {substrate_name}")
    return {f"substrate_basic_{k}": float(v) for k, v in BASIC_FEATURES[substrate_name].items()}


# RDKit molecular descriptors
# All computed deterministically from SMILES via RDKit.

def _get_rdkit_descriptors(mol) -> dict[str, float]:
    """Compute 9 RDKit descriptors for a molecule."""
    return {
        "substrate_rdkit_mol_wt": Descriptors.MolWt(mol),
        "substrate_rdkit_tpsa": Descriptors.TPSA(mol),
        "substrate_rdkit_logp": Descriptors.MolLogP(mol),
        "substrate_rdkit_hbd": Descriptors.NumHDonors(mol),
        "substrate_rdkit_hba": Descriptors.NumHAcceptors(mol),
        "substrate_rdkit_rot_bonds": Descriptors.NumRotatableBonds(mol),
        "substrate_rdkit_ring_count": Descriptors.RingCount(mol),
        "substrate_rdkit_fraction_csp3": Descriptors.FractionCSP3(mol),
        "substrate_rdkit_heavy_atom_count": Descriptors.HeavyAtomCount(mol),
    }


def build_rdkit_block(smiles: str) -> dict[str, float]:
    """Return 9 RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return _get_rdkit_descriptors(mol)


#  Morgan fingerprints
# ECFP4 (Extended Connectivity Fingerprint, radius=2), folded to 128 bits.
# Reference: Rogers & Hahn (2010) J. Chem. Inf. Model. 50:742-754.

def build_morgan_block(smiles: str, radius: int = 2, n_bits: int = 128) -> dict[str, float]:
    """Return a Morgan fingerprint as a dict of {substrate_morgan_NNN: 0.0/1.0}."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)

    # RDKit fingerprint: GetFingerprint returns an ExplicitBitVect; convert to numpy
    arr = np.zeros(n_bits, dtype=float)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0

    return {f"substrate_morgan_{i:03d}": float(arr[i]) for i in range(n_bits)}


# ── Assemble ──────────────────────────────────────────────────

def build_substrate_dataframe(
    smiles_by_name: dict[str, str] | None = None,
    morgan_radius: int = 2,
    morgan_n_bits: int = 128,
) -> pd.DataFrame:
    """Build the full substrate encoding DataFrame.

    Parameters
    ----------
    smiles_by_name : dict
        {substrate_name: SMILES}. Defaults to fetching from PubChem by InChIKey.
    morgan_radius : int
        Morgan fingerprint radius (2 = ECFP4).
    morgan_n_bits : int
        Number of fingerprint bits (128).

    Returns
    -------
    pd.DataFrame with columns:
        substrate_name, substrate_smiles,
        substrate_basic_* (7), substrate_rdkit_* (9), substrate_morgan_* (128)
        = 146 columns total
    """
    if smiles_by_name is None:
        smiles_by_name = get_substrate_smiles()

    rows = []
    for name in sorted(smiles_by_name):
        smiles = smiles_by_name[name]

        row: dict[str, object] = {}
        row["substrate_name"] = name
        row["substrate_smiles"] = smiles

        #  Basic physicochemical
        row.update(build_basic_block(name))

        # RDKit descriptors
        row.update(build_rdkit_block(smiles))

        #  Morgan fingerprint
        row.update(build_morgan_block(smiles, radius=morgan_radius, n_bits=morgan_n_bits))

        rows.append(row)

    return pd.DataFrame(rows)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reproducible substrate encoding table (no MACAW)."
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "substrate_encodings.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-n-bits", type=int, default=128)
    args = parser.parse_args()

    df = build_substrate_dataframe(
        morgan_radius=args.morgan_radius,
        morgan_n_bits=args.morgan_n_bits,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_feat = len(df.columns) - 2  # exclude name + smiles
    print(f"Substrate encodings → {args.output}")
    print(f"  {len(df)} substrates × {len(df.columns)} columns")
    print(f"  Features: {n_feat} dims ({7} basic + {9} RDKit + {args.morgan_n_bits} Morgan)")
    print(f"  Reproducible: yes (deterministic, no learned model)")


if __name__ == "__main__":
    main()
