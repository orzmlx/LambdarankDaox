# DAOx Substrate Preference Prediction by XGBoost with LambdaRank

Predict D-amino acid oxidase (DAOx) variant substrate preferences using ESM3 embeddings and learning-to-rank, implemented with **XGBoost** (`XGBRanker`, `objective='rank:pairwise'`).

| Mode | Variants | Prior | LambdaRank | Improvement |
|------|:---:|:---:|:---:|:---:|
| Curated (≥10 variants/position) | 2,073 | 71.4% | **73.4%** | +2.0pp |
| Full (all missense variants) | 5,730 | 70.3% | **72.1%** | +1.8pp |

All features are deterministically reproducible from public databases.

## Quick Start

```bash
pip install -r requirements.txt

# Regenerate substrate encodings (optional — pre-generated CSV included)
python3 scripts/build_substrate_encodings.py

# Train and evaluate
python3 lambdarank_ensemble.py              # Curated 2073 variants (default, best accuracy)
python3 lambdarank_ensemble.py --full       # Full 5730 variants
python3 lambdarank_ensemble.py --n-seeds 7  # 7-model ensemble
```

## Method

Each enzyme variant must rank 5 substrates (D-Ala, D-Asn, D-Gln, D-Met, D-Phe) by fitness. We treat this as a **learning-to-rank** problem:

1. For each `(variant, substrate)` pair, build a feature vector: `ESM3 PCA(128d) + substrate encoding(144d) + position prior(1d) = 273d`
   - ESM3 PCA 128d: 1536-dim mean-pooled embedding → PCA to 128
   - Substrate encoding 144d: 7 basic properties (side chain atoms, hydropathy, volume, charge, aromaticity, sulfur, amide) + 9 RDKit molecular descriptors (Molecular Weight, TPSA, logP, HBD/HBA, rotatable bonds, rings, sp3 fraction, heavy atoms) + 128 Morgan fingerprint (ECFP4, radius=2)
   - Position prior 1d: training-set mean fitness for this position × substrate pair
2. Train XGBoost XGBRanker with pairwise logistic loss — 5 substrates per variant form one ranking group
3. Ensemble 7 models with different random seeds


> **Note**: Docking features (~400 columns) exist in `pair_dataset.csv` but are **not used** by the model. Adding them consistently degrades performance (tested extensively). The 273-dim baseline is optimal.

## Data

| File | Description |
|------|-------------|
| `data/pair_dataset.csv` | 2,073 curated variants × 5 substrates, with ESM3 embeddings (1536d). Also contains docking columns (~400d) which are **not used** by the model. |
| `data/fitness_labels.csv` | 5,800 variants fitness data (used in `--full` mode with ESM3 embeddings) |
| `data/variant_esm3_embeddings.csv` | ESM3 embeddings for all 5,730 variants (`--full` mode) |
| `data/substrate_encodings.csv` | 144-dim features: 7 physicochemical + 9 RDKit + 128 Morgan FP |
| `data/source_data.xlsx` | Original NatComm 2026 source data (all figures + biological replicates) |
| `data/distance.csv` | CA-to-FAD N5 distances for all 360 positions (computed from PDB) |

## Scripts

| Script | Description |
|--------|-------------|
| `lambdarank_ensemble.py` | Main LambdaRank training + evaluation pipeline |
| `scripts/build_substrate_encodings.py` | Generate substrate_encodings.csv from SMILES via RDKit (deterministic) |
| `scripts/build_fitness_labels.py` | Extract fitness labels from `source_data.xlsx` |
| `scripts/build_all_data.py` | One-click: build both fitness_labels + substrate_encodings |
| `scripts/bio_data.py` | Reference data (BLOSUM62 from NCBI, AA properties from AAindex) |
| `scripts/cal_distance.py` | Compute per-residue CA-to-FAD distances from PDB structure |
| `scripts/pairwise_physicochemical.py` | Mutation effect features (dCharge/dVol/dHydro) |

## Requirements

- Python 3.10+
- xgboost, scikit-learn, numpy, pandas, scipy
- RDKit (`pip install rdkit`) — for `build_substrate_encodings.py`

## Reproducibility

All 273 feature dimensions are deterministic:
- ESM3 PCA128: fixed `random_state=42`
- Substrate encoding 144d: SMILES → PubChem (via InChIKey) → RDKit (deterministic)
- Position Prior: computed from training data only (fold-safe, per GroupKFold split)

Run `scripts/build_all_data.py` to regenerate fitness labels and substrate encodings from `source_data.xlsx`.
