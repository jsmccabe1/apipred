#!/usr/bin/env python3
"""
ApiPred - Apicomplexan protein function predictor
====================================================

Multi-species, compartment-aware, structurally contextualised predictions
for any apicomplexan proteome from ESM-2 embeddings.

For each protein, ApiPred predicts:
  - Essentiality (CRISPR fitness score)
  - Subcellular compartment
  - Invasion machinery membership
  - Structural novelty (distance from nearest known protein)
  - Top 3 most similar characterised proteins (with descriptions)

Usage:
    python predict.py --input proteome.fasta --output predictions.tsv
    python predict.py --input proteome.fasta --output predictions.tsv --device cuda
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_esm2_model():
    """Load ESM-2 650M model."""
    print("Loading ESM-2 model...")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def embed_protein(seq, model, alphabet, batch_converter, device="cpu",
                  layers=(20, 24, 28, 33), max_len=1022):
    """Generate protein-level ESM-2 embedding for a single sequence."""
    seq = seq.replace("*", "").replace("X", "A").replace("U", "C")
    seq = seq.replace("B", "N").replace("Z", "Q").replace("J", "L")
    if len(seq) == 0:
        return None

    if len(seq) <= max_len:
        windows = [seq]
    else:
        stride = max_len - 200
        windows = []
        for start in range(0, len(seq), stride):
            end = min(start + max_len, len(seq))
            windows.append(seq[start:end])
            if end == len(seq):
                break

    all_reps = []
    for window_seq in windows:
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", window_seq)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(layers), return_contacts=False)
        layer_reps = [results["representations"][l][0, 1:-1] for l in layers]
        mean_rep = torch.stack(layer_reps).mean(dim=0)
        all_reps.append(mean_rep.cpu())

    combined = torch.cat(all_reps, dim=0)
    return combined.mean(dim=0).numpy()


def parse_fasta(fasta_path):
    """Parse FASTA file."""
    sequences = []
    current_id = None
    current_desc = ""
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences.append((current_id, current_desc, "".join(current_seq)))
                parts = line[1:].split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        sequences.append((current_id, current_desc, "".join(current_seq)))
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="ApiPred: predict essential genes, compartments, and structural "
                    "context in apicomplexan proteomes using ESM-2 embeddings"
    )
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for ESM-2 (default: cpu)")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing pre-trained models")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "models"

    essentiality_path = model_dir / "essentiality_model.joblib"
    invasion_path = model_dir / "invasion_model.joblib"
    reference_path = model_dir / "reference_db.npz"

    has_models = essentiality_path.exists()
    has_reference = reference_path.exists()

    if not has_models:
        print(f"No pre-trained models found in {model_dir}/")
        print(f"Run: python train_model.py --data-dir /path/to/Apicomplexa/")
        print(f"Proceeding with embedding-only structural context analysis...\n")

    # Parse input
    print(f"Reading {args.input}...")
    sequences = parse_fasta(args.input)
    print(f"  {len(sequences)} proteins")
    if len(sequences) == 0:
        print("ERROR: No sequences found")
        sys.exit(1)

    # Embed
    model, alphabet, batch_converter = load_esm2_model()
    model = model.to(args.device)

    print(f"Embedding {len(sequences)} proteins...")
    embeddings = {}
    for i, (seq_id, desc, seq) in enumerate(sequences):
        emb = embed_protein(seq, model, alphabet, batch_converter, device=args.device)
        if emb is not None:
            embeddings[seq_id] = emb
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(sequences)}]")
    print(f"  Embedded: {len(embeddings)}")

    seq_ids = list(embeddings.keys())
    X = np.array([embeddings[sid] for sid in seq_ids])
    desc_map = {sid: desc for sid, desc, _ in sequences}
    len_map = {sid: len(seq) for sid, _, seq in sequences}

    # Predict essentiality and invasion
    if has_models:
        import joblib
        print("Loading prediction models...")
        ess_models = joblib.load(essentiality_path)
        predicted_scores = ess_models["regressor"].predict(X)
        essential_probs = ess_models["classifier"].predict_proba(X)[:, 1]

        if invasion_path.exists():
            inv_model = joblib.load(invasion_path)
            inv_probs = inv_model.predict_proba(X)[:, 1]
        else:
            inv_probs = np.full(len(X), np.nan)
    else:
        predicted_scores = np.full(len(X), np.nan)
        essential_probs = np.full(len(X), np.nan)
        inv_probs = np.full(len(X), np.nan)

    # Structural context from reference database
    if has_reference:
        print("Loading reference database...")
        ref_data = np.load(reference_path, allow_pickle=True)
        ref_embs = ref_data["embeddings"]
        ref_ids = ref_data["ids"]
        ref_descs = ref_data["descriptions"]
        ref_comps = ref_data["compartments"]
        ref_scores = ref_data["crispr_scores"]
        ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
        ref_normed = ref_embs / (ref_norms + 1e-10)
    else:
        ref_normed = None

    # Build results
    print("Generating predictions...")
    results = []
    for i, sid in enumerate(seq_ids):
        score = float(predicted_scores[i])
        ess_prob = float(essential_probs[i])
        inv_prob = float(inv_probs[i])

        if np.isnan(score):
            ess_class = "no_model"
        elif score < -3:
            ess_class = "essential"
        elif score < -1:
            ess_class = "important"
        else:
            ess_class = "dispensable"

        row = {
            "protein_id": sid,
            "description": desc_map.get(sid, "")[:80],
            "length": len_map.get(sid, 0),
            "predicted_crispr_score": round(score, 3) if not np.isnan(score) else "",
            "essential_probability": round(ess_prob, 3) if not np.isnan(ess_prob) else "",
            "essentiality_class": ess_class,
            "invasion_probability": round(inv_prob, 3) if not np.isnan(inv_prob) else "",
            "predicted_invasion": "yes" if inv_prob > 0.5 else ("no" if not np.isnan(inv_prob) else ""),
        }

        # Structural context
        if ref_normed is not None:
            query_normed = X[i] / (np.linalg.norm(X[i]) + 1e-10)
            sims = ref_normed @ query_normed
            top3_idx = np.argsort(sims)[-3:][::-1]

            for rank, j in enumerate(top3_idx, 1):
                row[f"similar_{rank}_id"] = str(ref_ids[j])
                row[f"similar_{rank}_desc"] = str(ref_descs[j])[:50]
                row[f"similar_{rank}_compartment"] = str(ref_comps[j])
                row[f"similar_{rank}_similarity"] = round(float(sims[j]), 4)
                ref_s = ref_scores[j]
                row[f"similar_{rank}_crispr"] = round(float(ref_s), 2) if np.isfinite(ref_s) else ""

            max_sim = float(sims[top3_idx[0]])
            row["max_similarity_to_known"] = round(max_sim, 4)
            row["structural_novelty"] = "novel" if max_sim < 0.95 else "known_fold"

        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("predicted_crispr_score" if has_models else "protein_id")
    df.to_csv(args.output, sep="\t", index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"ApiPred results: {args.output}")
    print(f"{'='*60}")
    print(f"  Proteins analysed:     {len(df)}")
    if has_models:
        n_ess = (df["essentiality_class"] == "essential").sum()
        n_imp = (df["essentiality_class"] == "important").sum()
        n_inv = (df["predicted_invasion"] == "yes").sum()
        print(f"  Predicted essential:   {n_ess}")
        print(f"  Predicted important:   {n_imp}")
        print(f"  Predicted invasion:    {n_inv}")
    if has_reference:
        n_novel = (df["structural_novelty"] == "novel").sum()
        print(f"  Structurally novel:    {n_novel}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
