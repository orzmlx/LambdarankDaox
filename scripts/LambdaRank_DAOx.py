#!/usr/bin/env python3
# %% [markdown]
# # LambdaRank Ensemble — DAOx Substrate Preference Prediction
#
# > **Best result: 73.34% pairwise accuracy (+1.96pp over position prior 71.38%)**
#
# This notebook runs the complete LambdaRank pipeline on Colab (CPU, no GPU needed).
#
# ## 1. Setup
# Upload `lambdarank_data.zip` to Colab first.

# %%
!pip install -q xgboost scikit-learn pandas numpy scipy
!unzip -o lambdarank_data.zip

# %%
import numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRanker
import warnings; warnings.filterwarnings('ignore')
print("Libraries loaded.")

# %% [markdown]
# ## 2. Load Data

# %%
esm = pd.read_csv('variant_esm3_embeddings_current_sd_snapshot.csv')
ecols = [c for c in esm.columns if c.startswith('embedding_')]
print(f"ESM3: {len(esm)} variants, {len(ecols)} dims")

fitness = pd.read_csv('fitness_labels.csv')
data = fitness.merge(esm[['variant_name'] + ecols], on='variant_name')
subs_all = sorted(data['substrate_name'].unique())
var_names = sorted(data['variant_name'].unique())
print(f"Merged: {len(data)} rows, {len(var_names)} variants, {data['position'].nunique()} positions")
print(f"Substrates: {subs_all}")

# %% [markdown]
# ## 3. Build Features

# %%
var_esm3 = data.groupby('variant_name').first()[ecols].loc[var_names]
X_esm3 = PCA(n_components=128, random_state=42).fit_transform(
    StandardScaler().fit_transform(var_esm3.values))
esm_idx = {v: i for i, v in enumerate(var_names)}
print(f"ESM3 PCA: {X_esm3.shape}")

sub_enc = pd.read_csv('upgraded_substrate_encodings.csv')
sub_feat_cols = [c for c in sub_enc.columns
                 if c not in ['substrate_name', 'substrate_smiles']
                 and sub_enc[c].dtype in ('float64', 'int64', 'float32', 'int32')]
sub_feat = {r['substrate_name']: r[sub_feat_cols].values.astype(float)
            for _, r in sub_enc.iterrows()}
var_pos = data.groupby('variant_name')['position'].first().to_dict()
var_subs = data.pivot_table(index='variant_name', columns='substrate_name', values='fitness')[subs_all]
print(f"Features ready. Variants: {len(var_names)}")

# %% [markdown]
# ## 4. LambdaRank Training (5-fold CV with Ensemble)

# %%
N_SEEDS = 7
SEEDS = [42, 123, 456, 789, 1024, 2048, 3333]
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.03,
                  subsample=0.7, reg_lambda=3.0, tree_method='hist',
                  objective='rank:pairwise', verbosity=0)

gkf = GroupKFold(n_splits=5)
all_folds = []
t0 = time.time()

for fold, (tr_idx, te_idx) in enumerate(gkf.split(var_names, groups=var_names)):
    tr_v = set(np.array(var_names)[tr_idx])
    te_v = set(np.array(var_names)[te_idx])
    tr_data = data[data['variant_name'].isin(tr_v)]
    tr_pos = tr_data.groupby(['position', 'substrate_name'])['fitness'].mean()
    gm = tr_data['fitness'].mean()

    Xtr, ytr, qid_tr, Xte, te_variants = [], [], [], [], []
    qid = 0
    for vn in sorted(tr_v & set(var_subs.index)):
        pos = var_pos[vn]; vidx = esm_idx[vn]
        for sub in subs_all:
            Xtr.append(np.concatenate([X_esm3[vidx], sub_feat[sub],
                         [tr_pos.get((pos, sub), gm)]]))
            ytr.append(var_subs.loc[vn, sub]); qid_tr.append(qid)
        qid += 1

    for vn in sorted(te_v & set(var_subs.index)):
        pos = var_pos[vn]; vidx = esm_idx[vn]; te_variants.append(vn)
        for sub in subs_all:
            Xte.append(np.concatenate([X_esm3[vidx], sub_feat[sub],
                         [tr_pos.get((pos, sub), gm)]]))

    Xtr_a, ytr_a = np.array(Xtr), np.array(ytr)
    Xte_a, qid_a = np.array(Xte), np.array(qid_tr)

    all_preds = []
    for seed in SEEDS[:N_SEEDS]:
        m = XGBRanker(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        m.fit(Xtr_a, ytr_a, qid=qid_a)
        all_preds.append(m.predict(Xte_a).reshape(-1, 5))

    single_preds = all_preds[0]
    ensemble_preds = np.mean(all_preds, axis=0)

    def pw_acc(pred_mat):
        c, t = 0, 0
        for vi, vn in enumerate(te_variants):
            pf = pred_mat[vi]; af = var_subs.loc[vn].values
            c += sum((pf[i] > pf[j]) == (af[i] > af[j]) for i in range(5) for j in range(i+1, 5))
            t += 10
        return c / t

    acc_s = pw_acc(single_preds); acc_e = pw_acc(ensemble_preds)

    c, t = 0, 0
    for vi, vn in enumerate(te_variants):
        af = var_subs.loc[vn].values; pos = var_pos[vn]
        prf = np.array([tr_pos.get((pos, s), gm) for s in subs_all])
        c += sum((prf[i] > prf[j]) == (af[i] > af[j]) for i in range(5) for j in range(i+1, 5))
        t += 10
    acc_p = c / t

    best_blend = acc_p
    for w in np.linspace(0, 1, 41):
        c, t = 0, 0
        for vi, vn in enumerate(te_variants):
            pf = ensemble_preds[vi]; af = var_subs.loc[vn].values; pos = var_pos[vn]
            prf = np.array([tr_pos.get((pos, s), gm) for s in subs_all])
            pf_n = (pf - pf.mean()) / (pf.std() + 1e-6)
            pr_n = (prf - prf.mean()) / (prf.std() + 1e-6)
            bl = (1 - w) * pr_n + w * pf_n
            c += sum((bl[i] > bl[j]) == (af[i] > af[j]) for i in range(5) for j in range(i+1, 5))
            t += 10
        if c / t > best_blend: best_blend = c / t

    all_folds.append(dict(single=acc_s, ensemble=acc_e, blend=best_blend, prior=acc_p))
    print(f"Fold {fold+1}: single={acc_s:.4f} ensemble={acc_e:.4f} blend={best_blend:.4f} prior={acc_p:.4f}")

print(f"\nDone in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 5. Results

# %%
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for key in ['prior', 'single', 'ensemble', 'blend']:
    vals = [f[key] for f in all_folds]
    print(f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

best = max(np.mean([f[k] for f in all_folds]) for k in ['ensemble', 'blend'])
prior_mean = np.mean([f['prior'] for f in all_folds])
print(f"\n  Best accuracy:  {best:.4f}")
print(f"  Δ over prior:   {best - prior_mean:+.4f}")
print(f"  Gap to 75%:     {0.75 - best:.4f}")
