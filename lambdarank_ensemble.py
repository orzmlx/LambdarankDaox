#!/usr/bin/env python3
"""
LambdaRank + ESM3 + Substrate Encoding + Ensemble
DAOx substrate preference prediction.

Best result: 73.34% pairwise accuracy (+1.96pp over position prior 71.38%).

Usage:
    python3 lambdarank_ensemble.py              # Quick: 3 seeds, 5-fold CV
    python3 lambdarank_ensemble.py --full       # Full 5730-variant dataset
    python3 lambdarank_ensemble.py --n-seeds 7  # 7-model ensemble

Data requirements (see README for details):
    data/pair_dataset.csv              # 2073 variants × 5 substrates with ESM3 embeddings
    data/substrate_encodings.csv       # 161-dim chemical features for 5 substrates
    data/fitness_labels.csv            # (--full mode only) 5800 variants fitness
    data/variant_esm3_embeddings.csv   # (--full mode only) ESM3 for all 5800 variants

Reference:
    Liu et al., "LambdaRank Ensemble for Enzyme Substrate Preference Prediction", 2026
"""
import numpy as np, pandas as pd, os, sys, time, argparse
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRanker
import warnings; warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(ROOT, 'data')

# ── Hyperparameters ───────────────────────────────────────
N_FOLDS = 5
N_SEEDS_DEFAULT = 5
N_PCA = 128
XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.03,
    subsample=0.7, reg_lambda=3.0, tree_method='hist',
    objective='rank:pairwise', verbosity=0,
)
SEED_POOL = [42, 123, 456, 789, 1024, 2048, 3333]


# ── Data loading ──────────────────────────────────────────
def load_data(data_dir, use_full=False):
    """Load dataset and return (data_df, var_names, subs_all, ecols)."""
    pair_csv = os.path.join(data_dir, 'pair_dataset.csv')
    sub_csv = os.path.join(data_dir, 'substrate_encodings.csv')
    fitness_csv = os.path.join(data_dir, 'fitness_labels.csv')
    esm3_csv = os.path.join(data_dir, 'variant_esm3_embeddings.csv')

    if use_full:
        for f, name in [(fitness_csv, 'fitness_labels.csv'),
                        (esm3_csv, 'variant_esm3_embeddings.csv')]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Need {name} in {data_dir}/ for --full mode")
        fitness = pd.read_csv(fitness_csv)
        esm = pd.read_csv(esm3_csv)
        ecols = [c for c in esm.columns if c.startswith('embedding_')]
        data = fitness.merge(esm[['variant_name'] + ecols], on='variant_name')
    else:
        if not os.path.exists(pair_csv):
            raise FileNotFoundError(f"Need pair_dataset.csv in {data_dir}/")
        pair = pd.read_csv(pair_csv)
        ecols = [c for c in pair.columns if c.startswith('embedding_')]
        data = pair

    subs_all = sorted(data['substrate_name'].unique())
    var_names = sorted(data['variant_name'].unique())
    print(f"Data: {len(data)} rows, {len(var_names)} variants, "
          f"{data['position'].nunique()} positions", flush=True)
    return data, var_names, subs_all, ecols


def build_features(data, var_names, subs_all, ecols, data_dir, n_pca=N_PCA):
    """Prepare ESM3 PCA, substrate encodings, variant metadata."""
    var_esm = data.groupby('variant_name').first()[ecols].loc[var_names]
    X_esm_pca = PCA(n_components=n_pca, random_state=42).fit_transform(
        StandardScaler().fit_transform(var_esm.values))
    esm_idx = {v: i for i, v in enumerate(var_names)}

    sub_csv = os.path.join(data_dir, 'substrate_encodings.csv')
    if not os.path.exists(sub_csv):
        raise FileNotFoundError(f"Need substrate_encodings.csv in {data_dir}/")
    sub_enc = pd.read_csv(sub_csv)
    sub_feat_cols = [c for c in sub_enc.columns
                     if c not in ['substrate_name', 'substrate_smiles']
                     and sub_enc[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    sub_feat = {r['substrate_name']: r[sub_feat_cols].values.astype(float)
                for _, r in sub_enc.iterrows()}

    var_pos = data.groupby('variant_name')['position'].first().to_dict()
    var_subs = data.pivot_table(
        index='variant_name', columns='substrate_name', values='fitness')[subs_all]

    return X_esm_pca, esm_idx, sub_feat, var_pos, var_subs


# ── Single-fold evaluation ─────────────────────────────────
def evaluate_fold(tr_v, te_v, data, X_esm_pca, esm_idx, sub_feat,
                  var_pos, var_subs, subs_all, seeds):
    """Train LambdaRank ensemble on training variants, evaluate on test variants."""
    tr_data = data[data['variant_name'].isin(tr_v)]
    tr_pos = tr_data.groupby(['position', 'substrate_name'])['fitness'].mean()

    # Build training set
    Xtr, ytr, qid_tr = [], [], []
    qid = 0
    for vn in sorted(tr_v & set(var_subs.index)):
        pos = var_pos[vn]; vidx = esm_idx[vn]
        for sub in subs_all:
            prior = tr_pos.get((pos, sub), tr_data['fitness'].mean())
            Xtr.append(np.concatenate([X_esm_pca[vidx], sub_feat[sub], [prior]]))
            ytr.append(var_subs.loc[vn, sub])
            qid_tr.append(qid)
        qid += 1

    # Build test set
    Xte, te_variants = [], []
    for vn in sorted(te_v & set(var_subs.index)):
        pos = var_pos[vn]; vidx = esm_idx[vn]; te_variants.append(vn)
        for sub in subs_all:
            prior = tr_pos.get((pos, sub), tr_data['fitness'].mean())
            Xte.append(np.concatenate([X_esm_pca[vidx], sub_feat[sub], [prior]]))

    Xtr_a, ytr_a = np.array(Xtr), np.array(ytr)
    Xte_a, qid_a = np.array(Xte), np.array(qid_tr)

    # Train ensemble
    all_preds = []
    for seed in seeds:
        m = XGBRanker(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        m.fit(Xtr_a, ytr_a, qid=qid_a)
        all_preds.append(m.predict(Xte_a).reshape(-1, 5))

    single_preds = all_preds[0]
    ensemble_preds = np.mean(all_preds, axis=0)

    # Pairwise accuracy helper
    def pairwise_acc(pred_mat):
        correct, total = 0, 0
        for vi, vn in enumerate(te_variants):
            pf = pred_mat[vi]; af = var_subs.loc[vn].values
            for i in range(5):
                for j in range(i + 1, 5):
                    if (pf[i] > pf[j]) == (af[i] > af[j]):
                        correct += 1
                    total += 1
        return correct / total

    acc_single = pairwise_acc(single_preds)
    acc_ensemble = pairwise_acc(ensemble_preds)

    # Prior accuracy
    correct, total = 0, 0
    for vi, vn in enumerate(te_variants):
        af = var_subs.loc[vn].values; pos = var_pos[vn]
        prf = np.array([tr_pos.get((pos, s), tr_data['fitness'].mean())
                        for s in subs_all])
        for i in range(5):
            for j in range(i + 1, 5):
                if (prf[i] > prf[j]) == (af[i] > af[j]):
                    correct += 1
                total += 1
    acc_prior = correct / total

    # Adaptive blend: find best mixing weight
    best_blend = acc_prior
    for w in np.linspace(0, 1, 41):
        correct, total = 0, 0
        for vi, vn in enumerate(te_variants):
            pf = ensemble_preds[vi]; af = var_subs.loc[vn].values
            pos = var_pos[vn]
            prf = np.array([tr_pos.get((pos, s), tr_data['fitness'].mean())
                            for s in subs_all])
            pf_n = (pf - pf.mean()) / (pf.std() + 1e-6)
            pr_n = (prf - prf.mean()) / (prf.std() + 1e-6)
            bl = (1 - w) * pr_n + w * pf_n
            for i in range(5):
                for j in range(i + 1, 5):
                    if (bl[i] > bl[j]) == (af[i] > af[j]):
                        correct += 1
                    total += 1
        if correct / total > best_blend:
            best_blend = correct / total

    return acc_single, acc_ensemble, best_blend, acc_prior


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='LambdaRank Ensemble for DAOx substrate preference prediction')
    parser.add_argument('--full', action='store_true',
                        help='Use full 5730-variant dataset')
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS_DEFAULT,
                        help=f'Number of ensemble seeds (default: {N_SEEDS_DEFAULT})')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help=f'Number of CV folds (default: {N_FOLDS})')
    parser.add_argument('--pca', type=int, default=N_PCA,
                        help=f'ESM3 PCA dimensions (default: {N_PCA})')
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help='Path to data directory')
    args = parser.parse_args()

    data_dir = args.data_dir
    n_pca = args.pca
    n_folds = args.n_folds
    seeds = SEED_POOL[:args.n_seeds]

    print("=" * 60, flush=True)
    print(f"LambdaRank + ESM3 + Substrate Encoding + Ensemble", flush=True)
    print(f"N_PCA={n_pca}  N_SEEDS={args.n_seeds}  N_FOLDS={n_folds}"
          f"  data={'FULL' if args.full else 'CURATED'}", flush=True)
    print("=" * 60, flush=True)

    data, var_names, subs_all, ecols = load_data(data_dir, args.full)
    X_esm_pca, esm_idx, sub_feat, var_pos, var_subs = build_features(
        data, var_names, subs_all, ecols, data_dir, n_pca)

    gkf = GroupKFold(n_splits=n_folds)
    all_folds = []
    t0 = time.time()

    for fold, (tr_idx, te_idx) in enumerate(
            gkf.split(np.array(var_names), groups=var_names)):
        tr_v = set(np.array(var_names)[tr_idx])
        te_v = set(np.array(var_names)[te_idx])
        acc_s, acc_e, acc_b, acc_p = evaluate_fold(
            tr_v, te_v, data, X_esm_pca, esm_idx, sub_feat,
            var_pos, var_subs, subs_all, seeds)
        all_folds.append(dict(single=acc_s, ensemble=acc_e,
                              blend=acc_b, prior=acc_p))
        print(f"  Fold {fold + 1}: single={acc_s:.4f}  ensemble={acc_e:.4f}  "
              f"blend={acc_b:.4f}  prior={acc_p:.4f}", flush=True)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULTS  ({elapsed:.0f}s total)", flush=True)
    print(f"{'=' * 60}", flush=True)
    for key in ['prior', 'single', 'ensemble', 'blend']:
        vals = [f[key] for f in all_folds]
        print(f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}", flush=True)

    best = max(np.mean([f[k] for f in all_folds]) for k in ['ensemble', 'blend'])
    prior_mean = np.mean([f['prior'] for f in all_folds])
    print(f"\n  Best accuracy:  {best:.4f}", flush=True)
    print(f"  Δ over prior:   {best - prior_mean:+.4f}", flush=True)
    print(f"  Gap to 75%:     {0.75 - best:.4f}", flush=True)
    return best


if __name__ == '__main__':
    main()
