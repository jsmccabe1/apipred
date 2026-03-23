#!/usr/bin/env python3
"""
Train ApiPred models from T. gondii data.

Generates three files in models/:
  - essentiality_model.joblib  (regressor + classifier for CRISPR scores)
  - invasion_model.joblib      (classifier for invasion compartment)
  - reference_db.npz           (embedding database for structural context)

Usage:
    python train_model.py --data-dir ~/Apicomplexa/

The --data-dir should contain:
  - results/embeddings/all_proteins/protein_embeddings.npy
  - results/embeddings/all_proteins/protein_ids.txt
  - data/processed/protein_features.tsv
  - data/processed/protein_compartments.tsv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
import joblib
import warnings
warnings.filterwarnings("ignore")

INVASION_COMPARTMENTS = {
    "rhoptries 1", "rhoptries 2", "micronemes", "dense granules",
    "IMC", "apical 1", "apical 2",
}


def main():
    parser = argparse.ArgumentParser(description="Train ApiPred models")
    parser.add_argument("--data-dir", required=True, help="Path to Apicomplexa project directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for models")
    parser.add_argument("--fast", action="store_true", help="Use fewer estimators for faster training")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_est = 100 if args.fast else 200
    max_d = 3 if args.fast else 4

    # Load data
    print("Loading T. gondii data...")
    emb = np.load(data_dir / "results/embeddings/all_proteins/protein_embeddings.npy")
    with open(data_dir / "results/embeddings/all_proteins/protein_ids.txt") as f:
        ids = [l.strip() for l in f]
    idx_map = {pid: i for i, pid in enumerate(ids)}

    feat_df = pd.read_csv(data_dir / "data/processed/protein_features.tsv", sep="\t")
    comp_df = pd.read_csv(data_dir / "data/processed/protein_compartments.tsv", sep="\t")
    comp_dict = dict(zip(comp_df["accession"], comp_df["compartment"]))
    desc_dict = dict(zip(comp_df["accession"], comp_df.get("description", pd.Series(dtype=str))))

    print(f"  {len(ids)} proteins, {len(feat_df)} with features, {len(comp_df)} with compartments")

    # ── ESSENTIALITY MODEL ──
    print("\n--- Training essentiality models ---")
    ess_rows = []
    for _, r in feat_df.iterrows():
        acc = r["Accession"]
        score = r["CRISPR.Score"]
        if acc in idx_map and pd.notna(score):
            ess_rows.append({"accession": acc, "crispr_score": score, "idx": idx_map[acc]})
    ess_df = pd.DataFrame(ess_rows)

    X_ess = emb[ess_df["idx"].values]
    y_cont = ess_df["crispr_score"].values
    y_bin = (y_cont < -3).astype(int)

    print(f"  Training set: {len(ess_df)} proteins ({y_bin.sum()} essential)")

    regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=0.05,
            subsample=0.8, random_state=42
        ))
    ])

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=0.05,
            subsample=0.8, random_state=42
        ))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("  Training regressor (5-fold CV)...")
    y_pred = cross_val_predict(regressor, X_ess, y_cont, cv=cv)
    rho, _ = spearmanr(y_cont, y_pred)
    r2 = r2_score(y_cont, y_pred)
    print(f"    Spearman rho: {rho:.4f}, R²: {r2:.4f}")

    print("  Training classifier (5-fold CV)...")
    probs = cross_val_predict(classifier, X_ess, y_bin, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_bin, probs)
    print(f"    ROC AUC: {auc:.4f}")

    print("  Fitting final models on all data...")
    regressor.fit(X_ess, y_cont)
    classifier.fit(X_ess, y_bin)

    joblib.dump({"regressor": regressor, "classifier": classifier},
                output_dir / "essentiality_model.joblib")
    print(f"  Saved: {output_dir / 'essentiality_model.joblib'}")

    # ── INVASION MODEL ──
    print("\n--- Training invasion model ---")
    inv_rows = []
    for pid in ids:
        comp = comp_dict.get(pid, "unknown")
        if comp != "unknown":
            inv_rows.append({"accession": pid, "is_invasion": int(comp in INVASION_COMPARTMENTS),
                             "idx": idx_map[pid]})
    inv_df = pd.DataFrame(inv_rows)

    X_inv = emb[inv_df["idx"].values]
    y_inv = inv_df["is_invasion"].values
    print(f"  Training set: {len(inv_df)} proteins ({y_inv.sum()} invasion)")

    inv_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=0.05,
            subsample=0.8, random_state=42
        ))
    ])

    inv_probs = cross_val_predict(inv_clf, X_inv, y_inv, cv=cv, method="predict_proba")[:, 1]
    inv_auc = roc_auc_score(y_inv, inv_probs)
    print(f"    ROC AUC: {inv_auc:.4f}")

    inv_clf.fit(X_inv, y_inv)
    joblib.dump(inv_clf, output_dir / "invasion_model.joblib")
    print(f"  Saved: {output_dir / 'invasion_model.joblib'}")

    # ── REFERENCE DATABASE ──
    print("\n--- Building reference database ---")
    # Include all proteins with compartment assignments
    ref_ids = []
    ref_descs = []
    ref_comps = []
    ref_scores = []
    ref_indices = []

    crispr_dict = dict(zip(feat_df["Accession"], feat_df["CRISPR.Score"]))

    for pid in ids:
        comp = comp_dict.get(pid)
        if comp and comp != "unknown" and pid in idx_map:
            ref_ids.append(pid)
            ref_descs.append(str(desc_dict.get(pid, ""))[:80])
            ref_comps.append(comp)
            score = crispr_dict.get(pid, np.nan)
            ref_scores.append(float(score) if pd.notna(score) else np.nan)
            ref_indices.append(idx_map[pid])

    ref_embs = emb[ref_indices]
    print(f"  Reference database: {len(ref_ids)} characterised proteins")

    np.savez(output_dir / "reference_db.npz",
             embeddings=ref_embs,
             ids=np.array(ref_ids, dtype=object),
             descriptions=np.array(ref_descs, dtype=object),
             compartments=np.array(ref_comps, dtype=object),
             crispr_scores=np.array(ref_scores, dtype=float))
    print(f"  Saved: {output_dir / 'reference_db.npz'}")

    print(f"\n{'='*60}")
    print(f"ApiPred model training complete")
    print(f"{'='*60}")
    print(f"  Essentiality: rho={rho:.3f}, AUC={auc:.3f}")
    print(f"  Invasion: AUC={inv_auc:.3f}")
    print(f"  Reference DB: {len(ref_ids)} proteins")
    print(f"  Models saved to: {output_dir}/")
    print(f"\n  To use: python predict.py --input proteome.fasta --output predictions.tsv")


if __name__ == "__main__":
    main()
