#!/usr/bin/env python3
"""
Pure physicochemical: position identity + mutation AA properties + substrate chemistry
+ mutation×substrate interaction features → pairwise preference prediction.
No ESM3, no docking. Fold-safe prior baseline.

AA_PROPS and BLOSUM62 are fetched from authoritative sources at import time:
  - BLOSUM62 → NCBI FTP (Henikoff & Henikoff 1992)
  - AA_PROPS → Published reference values (Kyte-Doolittle 1982, Creighton 1993, etc.)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow importing bio_data from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from bio_data import AA_PROPS, BLOSUM62

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── Load ──
fitness = pd.read_csv(DATA / "fitness_labels.csv")
sub_enc = pd.read_csv(DATA / "substrate_encodings.csv")
sub_feat_cols = [c for c in sub_enc.columns if c not in ['substrate_name','substrate_smiles'] and sub_enc[c].dtype in ['float64','int64','float32','int32']]
df = fitness.merge(sub_enc[['substrate_name']+sub_feat_cols], on='substrate_name')
subs = sorted(df['substrate_name'].unique())
print(f"Data: {len(df)} rows, {df.variant_name.nunique()} variants, {df.position.nunique()} positions")

# ── Mutation features ──
for k in ['h','v','c','p','f']:
    df[f'mut_{k}'] = df['mutant_residue'].map(lambda a: AA_PROPS.get(a,{}).get(k,0))
    df[f'wt_{k}']  = df['wt_residue'].map(lambda a: AA_PROPS.get(a,{}).get(k,0))
    df[f'd_{k}']   = df[f'mut_{k}'] - df[f'wt_{k}']
    df[f'ad_{k}']  = np.abs(df[f'd_{k}'])

df['blosum'] = df.apply(lambda r: BLOSUM62.get(r['wt_residue'],{}).get(r['mutant_residue'],0), axis=1)
df['cons'] = (df['blosum'] >= 0).astype(int)
df['rad'] = (df['blosum'] <= -2).astype(int)
df['charge_flip'] = (df['d_c'].abs() >= 1.5).astype(int)
df['same_charge_sign'] = (np.sign(df['mut_c']) == np.sign(df['wt_c'])).astype(int)

mut_cols = ['blosum','cons','rad','charge_flip','same_charge_sign']
for k in ['h','v','c','p','f']:
    mut_cols += [f'mut_{k}', f'd_{k}', f'ad_{k}']

# ── Legacy substrate features ──
legacy = ['side_chain_atoms','side_chain_hetero_atoms','is_aromatic','has_sulfur','has_amide','hydropathy','side_chain_volume']
legacy = [c for c in legacy if c in df.columns]

# ── Mutation × substrate interaction ──
for k in ['h','v','c']:
    for sc in legacy:
        col = f'ix_{k}_{sc}'
        df[col] = df[f'd_{k}'] * df[sc]
ix_cols = [c for c in df.columns if c.startswith('ix_')]

# ── Per-position substrate preference entropy ──
pos_ent = {}
for pos, g in df.groupby('position'):
    m = {s: float(g[g['substrate_name']==s]['fitness'].mean()) for s in subs}
    mv = np.clip(np.array(list(m.values())) + 1e-8, 1e-8, None)
    mv = mv / mv.sum()
    pos_ent[pos] = -np.sum(mv * np.log(mv))
    pos_ent[pos] /= np.log(len(subs))  # normalize
df['pos_ent'] = df['position'].map(pos_ent)
df['pos_n'] = df.groupby('position')['variant_name'].transform('nunique')

# ── Position one-hot ──
pos_oh = pd.get_dummies(df['position'].astype(str), prefix='p')
pos_cols = list(pos_oh.columns)

# ── Build per-variant, per-substrate feature matrix ──
all_feat_cols = mut_cols + sub_feat_cols + legacy + ix_cols + ['pos_ent','pos_n'] + pos_cols
all_feat_cols = [c for c in all_feat_cols if c in df.columns or c in pos_oh.columns]

# Merge OH
feat_mat = pd.concat([df[['variant_name','substrate_name','fitness','position'] + [c for c in all_feat_cols if c in df.columns and c not in pos_cols]].reset_index(drop=True), pos_oh.reset_index(drop=True)], axis=1)

# ── Configs ──
configs = [
    ('mut_sub', mut_cols + sub_feat_cols),
    ('mut_sub_ix', mut_cols + sub_feat_cols + ix_cols),
    ('mut_sub_pos', mut_cols + sub_feat_cols + pos_cols),
    ('mut_sub_pos_ix', mut_cols + sub_feat_cols + pos_cols + ix_cols + ['pos_ent','pos_n']),
]

def run_config(name, feat_list, model_fn, n_splits=5):
    valid = [c for c in feat_list if c in feat_mat.columns]
    gkf = GroupKFold(n_splits=n_splits)
    y = feat_mat['fitness'].values
    groups = feat_mat['variant_name'].values
    fold_p, fold_m, fold_b = [], [], []
    
    for fold, (tr, te) in enumerate(gkf.split(feat_mat, y, groups=groups), 1):
        tr_df = feat_mat.iloc[tr].copy()
        te_df = feat_mat.iloc[te].copy()
        
        # Fold-safe prior
        pos_mean = {}
        for pos, g in tr_df.groupby('position'):
            pos_mean[pos] = {s: float(g[g['substrate_name']==s]['fitness'].mean()) for s in subs}
        gm2 = {s: float(tr_df[tr_df['substrate_name']==s]['fitness'].mean()) for s in subs}
        tr_df['prior'] = tr_df.apply(lambda r: pos_mean.get(int(r['position']),{}).get(str(r['substrate_name']),gm2.get(str(r['substrate_name']),0)), axis=1)
        te_df['prior'] = te_df.apply(lambda r: pos_mean.get(int(r['position']),{}).get(str(r['substrate_name']),gm2.get(str(r['substrate_name']),0)), axis=1)
        
        # Build pairwise
        def mkpair(dd):
            rows = []
            for v, g in dd.groupby('variant_name'):
                recs = g.to_dict('records')
                for i in range(len(recs)):
                    for j in range(i+1, len(recs)):
                        L, R = recs[i], recs[j]
                        ex = {}
                        for c in valid + ['prior']:
                            if c in L and c in R:
                                ex[f'd_{c}'] = float(L[c]) - float(R[c])
                        ex['label'] = 1 if L['fitness'] > R['fitness'] else 0
                        rows.append(ex)
                        m = ex.copy(); m['label'] = 1 - ex['label']
                        for k in list(ex.keys()):
                            if k.startswith('d_'): m[k] = -ex[k]
                        rows.append(m)
            return pd.DataFrame(rows)
        
        trp = mkpair(tr_df)
        tep = mkpair(te_df)
        if len(trp) == 0 or len(tep) == 0:
            continue
        
        fc = [c for c in trp.columns if c != 'label']
        for c in fc:
            if c not in tep.columns: tep[c] = 0.0
        
        pp = (tep['d_prior'].values > 0).astype(int)
        pa = accuracy_score(tep['label'].values, pp)
        
        X_tr = trp[fc].fillna(0).values; y_tr = trp['label'].values
        X_te = tep[fc].fillna(0).values; y_te = tep['label'].values
        
        model = model_fn(42 + fold)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_te).astype(float)
        mp_pred = (prob > 0.5).astype(int)
        ma = accuracy_score(y_te, mp_pred)
        
        pdiff = tep['d_prior'].values
        ba = pa
        for mt in np.linspace(0, 0.20, 21):
            bl = np.where(np.abs(pdiff) >= mt, pp, mp_pred)
            ba = max(ba, accuracy_score(y_te, bl))
        
        fold_p.append(pa); fold_m.append(ma); fold_b.append(ba)
    
    if fold_p:
        return float(np.mean(fold_p)), float(np.mean(fold_m)), float(np.mean(fold_b)), len(valid)
    return None

print(f"\n{'='*80}")
print(f"{'Config':30s} | {'Model':16s} | {'prior':>6s} | {'model':>6s} | {'blend':>6s} | {'Δ':>7s} | {'n_feat':>6s}")
print(f"{'-'*80}")

t0 = time.time()
results = []
for cfg_name, feats in configs:
    for mname, mfn in [('xgb_shallow', lambda s: XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=2, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=s, n_jobs=-1, verbosity=0)),
                        ('xgb_deep', lambda s: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=2, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=s, n_jobs=-1, verbosity=0)),
                        ('rf', lambda s: RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=s, n_jobs=-1))]:
        if mname == 'rf' and len(feats) > 600:
            continue
        ret = run_config(cfg_name, feats, mfn)
        if ret:
            mp, mm, mb, nf = ret
            delta = mm - mp
            star = ' ★' if delta > 0.005 else ''
            print(f"{cfg_name:30s} | {mname:16s} | {mp:6.4f} | {mm:6.4f} | {mb:6.4f} | {delta:+7.4f}{star} | {nf:6d}")
            results.append((cfg_name, mname, mp, mm, mb, delta, nf))

print(f"\nTime: {time.time()-t0:.0f}s")
print(f"\nTop 10 by blend accuracy:")
results.sort(key=lambda r: r[4], reverse=True)
for i, (cfg, mn, mp, mm, mb, delta, nf) in enumerate(results[:10]):
    print(f"  {i+1}. {cfg:30s} {mn:16s} blend={mb:.4f} model={mm:.4f} prior={mp:.4f} Δ={delta:+.4f} ({nf}f)")
