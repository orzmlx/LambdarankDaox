#!/usr/bin/env python3
"""
Build Interaction Fingerprint (IFP) features from docking poses.

For each (variant, substrate) pair, extracts per-residue contact counts
from all available docking poses. Aggregates with median.

Usage:
    python3 scripts/build_ifp_features.py           # full run (~30 min)
    python3 scripts/build_ifp_features.py --quick   # 100 pairs for testing
"""

from __future__ import annotations
import sys, os, glob, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parent.parent
POSE_DIR = Path("/Users/liuxi/Desktop/MyAdventureStory/Yitian_Zhou_Interview/pdb/top_pair_poses")
VAR_DIR = Path("/Users/liuxi/Desktop/MyAdventureStory/Yitian_Zhou_Interview/pdb/variants")
DATA = ROOT / "data"

# ── Load position map ──
pmap = pd.read_csv(DATA / "pdb" / "daox_position_map.csv")
pdb_to_paper = dict(zip(pmap["pdb_residue_number"], pmap["source_position"]))

# Key residues: catalytic, substrate-contacting (from paper + structure)
KEY_PDB = [1052, 1054, 1056, 1058, 1213, 1223, 1225, 1238, 1285, 1335, 1336, 1339]
KEY_PAPER = sorted(set(pdb_to_paper.get(r, r) for r in KEY_PDB))


def parse_atoms(path: Path) -> list:
    atoms = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            resseq = int(line[22:26].strip())
            atoms.append((x, y, z, resseq))
    return atoms


def compute_ifp(ligand_pdb: Path, receptor_pdb: Path, cutoff: float = 4.0) -> dict:
    lig = parse_atoms(ligand_pdb)
    rec = parse_atoms(receptor_pdb)
    if not lig or not rec: return {}
    rec_coords = np.array([r[:3] for r in rec])
    tree = KDTree(rec_coords)
    ifp = {p: 0 for p in KEY_PAPER}
    for lx, ly, lz, _, in lig:
        idxs = tree.query_ball_point([lx, ly, lz], cutoff)
        for idx in idxs:
            rx, ry, rz, resseq = rec[idx]
            paper_pos = pdb_to_paper.get(resseq, resseq)
            if paper_pos in ifp:
                ifp[paper_pos] += 1
    return ifp


def aggregate_poses(pose_dir: Path, vs_key: str, receptor_pdb: Path) -> dict | None:
    pfs = sorted(glob.glob(str(pose_dir / f"{vs_key}_top_pose*.pdb")))
    if len(pfs) < 2: return None
    all_vecs = []
    for pf in pfs:
        ifp = compute_ifp(Path(pf), receptor_pdb)
        all_vecs.append(np.array([ifp.get(p, 0) for p in KEY_PAPER]))
    if len(all_vecs) < 2: return None
    stacked = np.array(all_vecs)
    result = {"variant": vs_key.split("_")[0], "substrate": "_".join(vs_key.split("_")[1:]), "n_poses": len(pfs)}
    for i, p in enumerate(KEY_PAPER):
        result[f"ifp_{p}_median"] = np.median(stacked[:, i])
        result[f"ifp_{p}_iqr"] = float(np.subtract(*np.percentile(stacked[:, i], [75, 25])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "ifp_features.csv")
    args = parser.parse_args()

    # Find rich pairs
    variants = sorted(os.listdir(POSE_DIR))
    rich_pairs = {}
    for f in sorted(glob.glob(str(POSE_DIR / "*_top_pose*.pdb"))):
        parts = os.path.basename(f).split("_top_pose")
        if len(parts) != 2: continue
        vs_key = parts[0]
        rich_pairs.setdefault(vs_key, []).append(f)

    rich = {k: v for k, v in rich_pairs.items() if len(v) >= 3}
    print(f"Pairs with >=3 poses: {len(rich)}")

    pairs = list(rich.items())
    if args.quick:
        pairs = pairs[:100]

    results = []
    for i, (vs_key, _) in enumerate(pairs):
        vn = vs_key.split("_")[0]
        sub = "_".join(vs_key.split("_")[1:])
        receptor = VAR_DIR / vn / f"{vn}_model.pdb"
        if not receptor.exists(): continue
        agg = aggregate_poses(POSE_DIR, vs_key, receptor)
        if agg is not None:
            results.append(agg)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pairs)}...")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output} ({len(df)} rows × {len(df.columns)} cols)")
    if len(df) > 0:
        print(f"  Mean poses/pair: {df['n_poses'].mean():.1f}")

        # Quick test: direction prediction with IFP features
        fl = pd.read_csv(DATA / "fitness_labels.csv")
        subs = sorted(fl["substrate_name"].unique())
        pv = fl.pivot_table(index="variant_name", columns="substrate_name", values="fitness")
        pv["winner"] = [subs[i] for i in pv[subs].values.argmax(axis=1)]

        ifp_cols = [c for c in df.columns if c.startswith("ifp_")]
        df["winner"] = df["variant"].map(pv["winner"])
        df_sw = df[df["winner"] != "D-Phe"].dropna(subset=["winner"])
        if len(df_sw) >= 20:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_score

            X_ifp = StandardScaler().fit_transform(df_sw[ifp_cols].values.astype(float))
            y = df_sw["winner"].values
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            cv = StratifiedKFold(3, shuffle=True, random_state=42)
            s = cross_val_score(clf, X_ifp, y, cv=cv, scoring="accuracy")
            print(f"\nIFP direction prediction: {s.mean():.4f}±{s.std():.4f} (chance=25%)")
            print(f"  IFP features: {len(ifp_cols)} (10 key residues × median + IQR)")


if __name__ == "__main__":
    main()
