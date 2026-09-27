"""
Test CatBoost / XGBoost ensembling with LightGBM on saved 34-feature pairs.
"""

import os
import sys
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import MODEL_PATH
from src.features import FEATURE_NAMES
from src.postprocessing import resolve_mutual_exclusivity

def evaluate_macro_f05(val_probs, all_meta_val, val_gt_dict, threshold_range=np.arange(0.50, 0.94, 0.02)):
    scored_cands = [
        (all_meta_val[i][0], all_meta_val[i][1], float(val_probs[i]))
        for i in range(len(all_meta_val))
    ]
    best_f05 = 0.0
    best_t = 0.50
    for t in threshold_range:
        t = round(float(t), 2)
        matches = resolve_mutual_exclusivity(scored_cands, threshold=t)
        scores = []
        for s1_id, truth in val_gt_dict.items():
            preds = set(matches.get(s1_id, []))
            if len(truth) == 0:
                scores.append(1.0 if len(preds) == 0 else 0.0)
            else:
                if len(preds) == 0:
                    scores.append(0.0)
                else:
                    tp = len(truth & preds)
                    p = tp / len(preds)
                    r = tp / len(truth)
                    if p + r == 0:
                        scores.append(0.0)
                    else:
                        scores.append((1.25 * p * r) / (0.25 * p + r))
        f05 = float(np.mean(scores))
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
    return best_t, best_f05

def main():
    dataset_path = os.path.join(PROJECT_DIR, "data", "train_val_pairs_34feats.npz")
    print(f"Loading pre-extracted dataset from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    all_meta_val = data['meta_val']
    val_s1_ids = data['val_s1_ids']
    val_gt_mids = data['val_gt_mids']

    val_gt_dict = {
        s1_id: set(mids)
        for s1_id, mids in zip(val_s1_ids, val_gt_mids)
    }

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # 1. Load LightGBM predictions
    print(f"Loading LightGBM model from {MODEL_PATH}...")
    lgb_model = lgb.Booster(model_file=MODEL_PATH)
    lgb_val_probs = lgb_model.predict(X_val)
    t_lgb, f_lgb = evaluate_macro_f05(lgb_val_probs, all_meta_val, val_gt_dict)
    print(f"LightGBM alone: Best threshold = {t_lgb:.2f}, Macro F0.5 = {f_lgb:.4f}")

    # 2. Train CatBoost
    print("\nTraining CatBoostClassifier (500 iterations, depth=6)...")
    t0 = time.time()
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = min(n_neg / max(n_pos, 1), 20.0)
    
    cb = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric='Logloss',
        random_seed=42,
        thread_count=-1,
        verbose=100
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=100)
    print(f"CatBoost training finished in {time.time()-t0:.1f}s")
    
    cb_val_probs = cb.predict_proba(X_val)[:, 1]
    t_cb, f_cb = evaluate_macro_f05(cb_val_probs, all_meta_val, val_gt_dict)
    print(f"CatBoost alone: Best threshold = {t_cb:.2f}, Macro F0.5 = {f_cb:.4f}")

    # 3. Evaluate Blends
    for w in [0.3, 0.5, 0.7]:
        blend_probs = w * lgb_val_probs + (1 - w) * cb_val_probs
        t_b, f_b = evaluate_macro_f05(blend_probs, all_meta_val, val_gt_dict)
        print(f"Blend (LightGBM {w*100:.0f}% + CatBoost {(1-w)*100:.0f}%): Best threshold = {t_b:.2f}, Macro F0.5 = {f_b:.4f}")

if __name__ == "__main__":
    main()
