#!/usr/bin/env python3
"""
Proof-of-concept: High-quality AutoDock Vina docking for F58W (known D-Ala booster).

Compares Vina docking scores with experimental fitness to test whether
BETTER docking can predict substrate preference.

Protocol:
  1. F58W receptor: from {VAR}/F58W/F58W_model.pdb → prepare with meeko+ADT
  2. Ligands: 5 D-amino acid SMILES → 3D conformers → PDBQT
  3. Vina docking: exhaustiveness=32, num_modes=100 per substrate
  4. Extract best docking score per substrate
  5. Rank by score → compare with experimental fitness ranking (Kendall's τ)

Usage:
    python3 scripts/dock_vina_poc.py
"""

from __future__ import annotations
import sys, subprocess, tempfile, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAR_DIR = ROOT.parent / "DAOx" / "pdb" / "variants"
TMP = ROOT / "data" / "vina_dock"
TMP.mkdir(exist_ok=True)

# ── Substrate SMILES ───────────────────────────────────────────
SUB_SMILES = {
    "D-Ala": "N[C@@H](C)C(=O)O",
    "D-Asn": "C([C@H](C(=O)O)N)C(=O)N",
    "D-Gln": "C(CC(=O)N)[C@H](C(=O)O)N",
    "D-Met": "CSCC[C@H](C(=O)O)N",
    "D-Phe": "C1=CC=C(C=C1)C[C@H](C(=O)O)N",
}

# ── Experimental fitness (from fitness_labels.csv) ─────────────
fl = pd.read_csv(ROOT / "data" / "fitness_labels.csv")
f58w = fl[fl["variant_name"] == "F58W"]
exp_fitness = {row["substrate_name"]: row["fitness"] for _, row in f58w.iterrows()}
exp_rank = sorted(exp_fitness, key=exp_fitness.get, reverse=True)
print(f"F58W experimental fitness: {exp_fitness}")
print(f"Experimental ranking: {' > '.join(exp_rank)}")

# ── Prepare receptor ───────────────────────────────────────────
variant = "F58W"
receptor_pdb = VAR_DIR / variant / f"{variant}_model.pdb"
receptor_pdbqt = TMP / f"{variant}_receptor.pdbqt"

print(f"\nPreparing receptor from {receptor_pdb}...")
# meeko PDB→PDBQT conversion (--allow_bad_res for non-standard FAD cofactor)
MEEKO = "/opt/anaconda3/envs/fastapi/bin"
r = subprocess.run([
    f"{MEEKO}/mk_prepare_receptor.py", "-i", str(receptor_pdb), "-o", str(receptor_pdbqt),
    "--allow_bad_res",
], capture_output=True, text=True)
if r.returncode != 0:
    print(f"Receptor prep warning: {r.stderr[-200:]}")
if not receptor_pdbqt.exists():
    print("ERROR: Receptor PDBQT not created. Falling back to manual conversion.")
    # Try using obabel instead
    subprocess.run(["obabel", str(receptor_pdb), "-O", str(receptor_pdbqt), "--gen3d"], capture_output=True)

# Get binding site center from FAD N5
# Parse PDB to find FAD N5
with open(receptor_pdb) as f:
    for line in f:
        if line.startswith("HETATM") and "FAD" in line[17:20] and "N5" in line[12:16]:
            cx, cy, cz = float(line[30:38]), float(line[38:46]), float(line[46:54])
            print(f"FAD N5 at ({cx:.1f}, {cy:.1f}, {cz:.1f})")
            break

# ── Dock each substrate ─────────────────────────────────────────
results = []
for sub_name, smiles in SUB_SMILES.items():
    print(f"\n{'='*50}")
    print(f"Docking {sub_name}...")

    # Prepare ligand via meeko
    lig_pdbqt = TMP / f"{sub_name}.pdbqt"
    subprocess.run([
        "mk_prepare_ligand.py",
        "-i", str(TMP / f"{sub_name}.sdf" if (TMP / f"{sub_name}.sdf").exists() else "-"),
        "--smiles", smiles,
        "-o", str(lig_pdbqt),
    ], check=True, capture_output=True, input=smiles.encode() if not (TMP/f"{sub_name}.sdf").exists() else None)

    # Write SMILES to temp SDF first (meeko needs SDF input)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    sdf_path = TMP / f"{sub_name}.sdf"
    Chem.MolToMolFile(mol, str(sdf_path))

    # Convert SDF→PDBQT with meeko
    MEEKO = "/opt/anaconda3/envs/fastapi/bin"
    result = subprocess.run([
        f"{MEEKO}/mk_prepare_ligand.py", "-i", str(sdf_path), "-o", str(lig_pdbqt)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  meeko ligand prep failed: {result.stderr[:200]}")
        # Fallback: use obabel
        subprocess.run(["obabel", str(sdf_path), "-O", str(lig_pdbqt), "--gen3d"], capture_output=True)
        if not lig_pdbqt.exists():
            continue

    # Run Vina
    out_pdbqt = TMP / f"{sub_name}_out.pdbqt"
    vina_cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(lig_pdbqt),
        "--out", str(out_pdbqt),
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", "20", "--size_y", "20", "--size_z", "20",
        "--exhaustiveness", "32",
        "--num_modes", "100",
        "--energy_range", "10",
    ]
    r = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=300)

    # Parse scores
    scores = []
    for line in r.stdout.split('\n'):
        if line.strip().startswith(('1 ','2 ','3 ','4 ','5 ','6 ','7 ','8 ','9 ')):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores.append(float(parts[1]))
                except: pass

    if scores:
        best = min(scores)  # Vina: more negative = better binding
        results.append({
            "substrate": sub_name,
            "best_score": best,
            "n_poses": len(scores),
            "mean_score": np.mean(scores),
            "exp_fitness": exp_fitness[sub_name],
        })
        print(f"  Best score: {best:.2f} kcal/mol ({len(scores)} poses)")
        print(f"  Pose scores: {sorted(scores)[:5]}...")
    else:
        print(f"  Docking failed for {sub_name}")

# ── Evaluate ────────────────────────────────────────────────────
if len(results) >= 3:
    df = pd.DataFrame(results)
    df = df.sort_values("best_score")  # best binding first
    print(f"\n{'='*60}")
    print("RESULTS: Vina Docking vs Experimental Fitness")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['substrate']:6s}: Vina={row['best_score']:.2f} kcal/mol, Exp fitness={row['exp_fitness']:.4f}")

    # Kendall's τ
    from scipy.stats import kendalltau
    tau, p = kendalltau(df["best_score"], df["exp_fitness"])
    print(f"\n  Kendall's τ = {tau:.4f} (p={p:.4f})")
    print(f"  → Docking {'CAN' if tau>0.1 else 'CANNOT'} predict experimental fitness ranking (τ={tau:.2f})")
    print(f"\n  D3R best: τ=0.37. Our benchmark: τ={tau:.2f}")

    df.to_csv(ROOT / "data" / "vina_poc_results.csv", index=False)
