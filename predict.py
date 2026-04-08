#!/usr/bin/env python3
"""
ApiPred - Apicomplexan protein function predictor
====================================================

Predicts essentiality, subcellular compartment, invasion machinery membership,
and structural context for any apicomplexan proteome from ESM-2 embeddings.

For each protein, ApiPred predicts:
  - Essentiality (CRISPR fitness score + confidence interval)
  - Subcellular compartment (multi-class: 7 invasion + 19 non-invasion)
  - Invasion machinery membership (with compartment-specific probability)
  - Structural novelty (distance from nearest characterised protein)
  - Top 3 most similar characterised proteins (with descriptions)

Usage:
    python predict.py --input proteome.fasta --output predictions.tsv
    python predict.py --input proteome.fasta --output predictions.tsv --device cuda
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
INVASION_COMPARTMENTS = {
    "rhoptries 1", "rhoptries 2", "micronemes", "dense granules",
    "IMC", "apical 1", "apical 2",
}

# Keywords identifying parasitism-specific invasion proteins (ROPKs, MICs, etc.)
# vs conserved alveolate proteins that happen to reside in invasion compartments
PARASITE_SPECIFIC_KEYWORDS = [
    "rhoptry protein rop", "rhoptry kinase", "ropk",
    "dense granule protein gra", "dense granule protein dg",
    "microneme protein mic", "microneme protein",
    "ama1", "apical membrane antigen",
    "rhoptry neck protein ron", "ron ",
    "toxofilin", "perforin", "saga", "sag1", "sag2",
    "surface antigen",
]
CONSERVED_ALVEOLATE_KEYWORDS = [
    "calmodulin", "centrin", "myosin", "tubulin", "alveolin",
    "hsp", "actin", "histone", "ribosom", "proteasome",
    "ubiquitin", "ef hand", "ef-hand", "calcium-binding",
    "kinase", "phosphatase", "atpase", "cyclophilin",
    "thioredoxin", "trx", "enolase", "aldolase",
]


def classify_match_specificity(description):
    """Classify whether a T. gondii match is parasite-specific or conserved."""
    d = str(description).lower()
    if any(kw in d for kw in PARASITE_SPECIFIC_KEYWORDS):
        return "parasite_specific"
    if any(kw in d for kw in CONSERVED_ALVEOLATE_KEYWORDS):
        return "conserved"
    if "hypothetical" in d or d in ("", "nan", "?"):
        return "unknown"
    return "unclassified"


def load_esm2_model():
    """Load ESM-2 650M model via the esm package."""
    try:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    except ImportError:
        model, alphabet = torch.hub.load("facebookresearch/esm:main",
                                         "esm2_t33_650M_UR50D")
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def clean_sequence(seq):
    """Clean protein sequence: replace non-standard AAs, remove stops."""
    seq = seq.upper().replace("*", "")
    return "".join(c if c in VALID_AA else "X" for c in seq)


def embed_batch(batch_seqs, model, alphabet, batch_converter, device="cpu",
                layers=(20, 24, 28, 33), max_len=1022):
    """Embed a batch of (id, seq) tuples. Returns dict of id -> embedding."""
    results = {}
    for seq_id, seq in batch_seqs:
        seq = clean_sequence(seq)
        if len(seq) == 0:
            continue

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
            data = [("protein", window_seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                out = model(batch_tokens, repr_layers=list(layers),
                            return_contacts=False)
            layer_reps = [out["representations"][l][0, 1:-1] for l in layers]
            mean_rep = torch.stack(layer_reps).mean(dim=0)
            all_reps.append(mean_rep.cpu())

        combined = torch.cat(all_reps, dim=0)
        results[seq_id] = combined.mean(dim=0).numpy()
    return results


def embed_proteome(sequences, model, alphabet, batch_converter,
                   device="cpu", batch_size=8):
    """Embed all sequences with batched processing and progress reporting."""
    # Sort by length for efficient GPU batching (similar lengths together)
    indexed = [(i, sid, seq) for i, (sid, _, seq) in enumerate(sequences)]
    indexed.sort(key=lambda x: len(x[2]))

    embeddings = {}
    n_total = len(indexed)
    n_done = 0
    n_failed = 0
    t0 = time.time()

    # Process in batches
    batch = []
    for _, sid, seq in indexed:
        if len(seq) < 10:
            n_failed += 1
            continue
        batch.append((sid, seq))

        if len(batch) >= batch_size:
            try:
                batch_embs = embed_batch(batch, model, alphabet,
                                         batch_converter, device)
                embeddings.update(batch_embs)
            except RuntimeError:
                # OOM - fall back to one at a time
                torch.cuda.empty_cache() if device == "cuda" else None
                for item in batch:
                    try:
                        single_emb = embed_batch([item], model, alphabet,
                                                 batch_converter, device)
                        embeddings.update(single_emb)
                    except RuntimeError:
                        n_failed += 1
                        torch.cuda.empty_cache() if device == "cuda" else None
            n_done += len(batch)
            batch = []

            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"\r  [{n_done}/{n_total}] {rate:.1f} seq/s "
                  f"ETA {eta:.0f}s | embedded: {len(embeddings)}", end="",
                  flush=True)

    # Process remainder
    if batch:
        try:
            batch_embs = embed_batch(batch, model, alphabet,
                                     batch_converter, device)
            embeddings.update(batch_embs)
        except RuntimeError:
            torch.cuda.empty_cache() if device == "cuda" else None
            for item in batch:
                try:
                    single_emb = embed_batch([item], model, alphabet,
                                             batch_converter, device)
                    embeddings.update(single_emb)
                except RuntimeError:
                    n_failed += 1

    elapsed = time.time() - t0
    print(f"\r  Embedded {len(embeddings)}/{n_total} proteins "
          f"in {elapsed:.1f}s ({n_failed} failed)     ")
    return embeddings


def parse_fasta(fasta_path):
    """Parse FASTA file with validation."""
    sequences = []
    current_id = None
    current_desc = ""
    current_seq = []
    n_warnings = 0

    with open(fasta_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    seq = "".join(current_seq)
                    if len(seq) >= 10:
                        sequences.append((current_id, current_desc, seq))
                    elif n_warnings < 5:
                        print(f"  Warning: {current_id} too short "
                              f"({len(seq)} aa), skipping")
                        n_warnings += 1
                parts = line[1:].split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)

    if current_id:
        seq = "".join(current_seq)
        if len(seq) >= 10:
            sequences.append((current_id, current_desc, seq))

    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="ApiPred: predict essential genes, compartments, and "
                    "structural context in apicomplexan proteomes"
    )
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for ESM-2 (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding (default: 8)")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing pre-trained models")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "models"

    essentiality_path = model_dir / "essentiality_ensemble.joblib"
    compartment_path = model_dir / "compartment_model.joblib"
    invasion_path = model_dir / "invasion_model.joblib"
    reference_path = model_dir / "reference_db.npz"

    # Backwards compatibility: check for old model names
    if not essentiality_path.exists() and (model_dir / "essentiality_model.joblib").exists():
        essentiality_path = model_dir / "essentiality_model.joblib"

    has_essentiality = essentiality_path.exists()
    has_compartment = compartment_path.exists()
    has_invasion = invasion_path.exists()
    has_reference = reference_path.exists()

    if not any([has_essentiality, has_compartment, has_invasion]):
        print(f"No pre-trained models found in {model_dir}/")
        print(f"Run: python train_model.py --data-dir /path/to/Apicomplexa/")
        print(f"Proceeding with embedding + structural context only...\n")

    # Parse input
    print(f"Reading {args.input}...")
    sequences = parse_fasta(args.input)
    print(f"  {len(sequences)} proteins")
    if len(sequences) == 0:
        print("ERROR: No valid sequences found")
        sys.exit(1)

    # Embed
    print("Loading ESM-2...")
    model, alphabet, batch_converter = load_esm2_model()
    model = model.to(args.device)

    print(f"Embedding {len(sequences)} proteins (batch_size={args.batch_size})...")
    embeddings = embed_proteome(sequences, model, alphabet, batch_converter,
                                device=args.device, batch_size=args.batch_size)

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if args.device == "cuda" else None

    seq_ids = [sid for sid, _, _ in sequences if sid in embeddings]
    X = np.array([embeddings[sid] for sid in seq_ids])
    desc_map = {sid: desc for sid, desc, _ in sequences}
    len_map = {sid: len(seq) for sid, _, seq in sequences}

    # ─── Essentiality prediction (ensemble for confidence) ───
    if has_essentiality:
        import joblib
        print("Loading essentiality model...")
        ess_data = joblib.load(essentiality_path)

        if "ensemble" in ess_data:
            # New ensemble format
            ensemble_preds = []
            ensemble_probs = []
            for fold_model in ess_data["ensemble"]:
                reg = fold_model["regressor"]
                clf = fold_model["classifier"]
                ensemble_preds.append(reg.predict(X))
                ensemble_probs.append(clf.predict_proba(X)[:, 1])

            predicted_scores = np.mean(ensemble_preds, axis=0)
            score_std = np.std(ensemble_preds, axis=0)
            essential_probs = np.mean(ensemble_probs, axis=0)
            prob_std = np.std(ensemble_probs, axis=0)
        else:
            # Old single-model format (backwards compatible)
            predicted_scores = ess_data["regressor"].predict(X)
            score_std = np.full(len(X), np.nan)
            essential_probs = ess_data["classifier"].predict_proba(X)[:, 1]
            prob_std = np.full(len(X), np.nan)
    else:
        predicted_scores = np.full(len(X), np.nan)
        score_std = np.full(len(X), np.nan)
        essential_probs = np.full(len(X), np.nan)
        prob_std = np.full(len(X), np.nan)

    # ─── Compartment prediction (multi-class) ───
    if has_compartment:
        import joblib
        print("Loading compartment model...")
        comp_data = joblib.load(compartment_path)
        comp_model = comp_data["model"]
        comp_classes = comp_data["classes"]

        comp_probs_all = comp_model.predict_proba(X)
        predicted_compartments = comp_classes[np.argmax(comp_probs_all, axis=1)]
        comp_confidence = np.max(comp_probs_all, axis=1)

        # Invasion probability = sum of invasion compartment probabilities
        inv_mask = np.array([c in INVASION_COMPARTMENTS for c in comp_classes])
        inv_probs = comp_probs_all[:, inv_mask].sum(axis=1)
    elif has_invasion:
        import joblib
        print("Loading invasion model (legacy)...")
        inv_model = joblib.load(invasion_path)
        inv_probs = inv_model.predict_proba(X)[:, 1]
        predicted_compartments = np.where(inv_probs > 0.5, "invasion", "non-invasion")
        comp_confidence = np.abs(inv_probs - 0.5) * 2
    else:
        inv_probs = np.full(len(X), np.nan)
        predicted_compartments = np.full(len(X), "", dtype=object)
        comp_confidence = np.full(len(X), np.nan)

    # ─── Structural context from reference database ───
    ref_context = {}
    contrastive_scores = np.full(len(X), np.nan)
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

        # Split reference into invasion vs non-invasion
        ref_inv_mask = np.array([c in INVASION_COMPARTMENTS for c in ref_comps])
        ref_noninv_mask = ~ref_inv_mask
        inv_normed = ref_normed[ref_inv_mask]
        noninv_normed = ref_normed[ref_noninv_mask]

        # Batch cosine similarity
        X_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normed = X / (X_norms + 1e-10)
        sim_matrix = X_normed @ ref_normed.T  # (n_query, n_ref)

        # Contrastive scoring: max similarity to invasion minus
        # max similarity to non-invasion. Positive = specifically
        # invasion-like beyond general structural similarity.
        inv_sims = X_normed @ inv_normed.T      # (n_query, n_inv)
        noninv_sims = X_normed @ noninv_normed.T  # (n_query, n_noninv)
        max_inv_sim = inv_sims.max(axis=1)
        max_noninv_sim = noninv_sims.max(axis=1)
        contrastive_scores = max_inv_sim - max_noninv_sim

        for i, sid in enumerate(seq_ids):
            sims = sim_matrix[i]
            top3_idx = np.argsort(sims)[-3:][::-1]
            context = {}
            for rank, j in enumerate(top3_idx, 1):
                context[f"similar_{rank}_id"] = str(ref_ids[j])
                context[f"similar_{rank}_desc"] = str(ref_descs[j])[:50]
                context[f"similar_{rank}_compartment"] = str(ref_comps[j])
                context[f"similar_{rank}_similarity"] = round(float(sims[j]), 4)
                ref_s = ref_scores[j]
                context[f"similar_{rank}_crispr"] = (
                    round(float(ref_s), 2) if np.isfinite(ref_s) else "")
            context["max_similarity_to_known"] = round(float(sims[top3_idx[0]]), 4)
            context["structural_novelty"] = (
                "novel" if sims[top3_idx[0]] < 0.95 else "known_fold")
            ref_context[sid] = context

    # ─── Build results ───
    print("Generating predictions...")
    results = []
    for i, sid in enumerate(seq_ids):
        score = float(predicted_scores[i])
        std = float(score_std[i])
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

        # Confidence level based on ensemble agreement
        if not np.isnan(std):
            if std < 0.5:
                confidence = "high"
            elif std < 1.0:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = ""

        row = {
            "protein_id": sid,
            "description": desc_map.get(sid, "")[:80],
            "length": len_map.get(sid, 0),
            "predicted_crispr_score": round(score, 3) if not np.isnan(score) else "",
            "score_std": round(std, 3) if not np.isnan(std) else "",
            "essential_probability": round(ess_prob, 3) if not np.isnan(ess_prob) else "",
            "essentiality_class": ess_class,
            "essentiality_confidence": confidence,
            "predicted_compartment": str(predicted_compartments[i]),
            "compartment_confidence": (round(float(comp_confidence[i]), 3)
                                       if not np.isnan(comp_confidence[i]) else ""),
            "invasion_probability": round(inv_prob, 3) if not np.isnan(inv_prob) else "",
            "contrastive_score": (round(float(contrastive_scores[i]), 4)
                                  if not np.isnan(contrastive_scores[i]) else ""),
            "predicted_invasion": ("yes" if (inv_prob > 0.5
                                             and contrastive_scores[i] > 0)
                                   else ("no" if not np.isnan(inv_prob) else "")),
        }

        # Structural context + specificity
        if sid in ref_context:
            row.update(ref_context[sid])
            # Classify whether the top match is parasite-specific or conserved
            top_desc = ref_context[sid].get("similar_1_desc", "")
            top_comp = ref_context[sid].get("similar_1_compartment", "")
            match_type = classify_match_specificity(top_desc)
            row["match_specificity"] = match_type

            # Invasion-specific flag: only "yes" if the protein matches a
            # parasite-specific invasion protein, not a conserved alveolate
            # protein that happens to be in an invasion compartment
            if inv_prob > 0.5 and match_type == "parasite_specific":
                row["invasion_specific"] = "yes"
            elif inv_prob > 0.5 and top_comp in INVASION_COMPARTMENTS:
                row["invasion_specific"] = "conserved_match"
            else:
                row["invasion_specific"] = "no"
        else:
            row["match_specificity"] = ""
            row["invasion_specific"] = ""

        results.append(row)

    df = pd.DataFrame(results)
    if has_essentiality:
        df = df.sort_values("predicted_crispr_score")
    df.to_csv(args.output, sep="\t", index=False)

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"ApiPred results: {args.output}")
    print(f"{'='*60}")
    print(f"  Proteins analysed:     {len(df)}")
    if has_essentiality:
        n_ess = (df["essentiality_class"] == "essential").sum()
        n_imp = (df["essentiality_class"] == "important").sum()
        n_hi = (df["essentiality_confidence"] == "high").sum()
        print(f"  Predicted essential:   {n_ess}")
        print(f"  Predicted important:   {n_imp}")
        print(f"  High-confidence calls: {n_hi}")
    if has_compartment or has_invasion:
        n_inv = (df["predicted_invasion"] == "yes").sum()
        n_specific = (df.get("invasion_specific") == "yes").sum()
        n_conserved = (df.get("invasion_specific") == "conserved_match").sum()
        print(f"  Predicted invasion (all):       {n_inv}")
        print(f"    Parasite-specific matches:    {n_specific}")
        print(f"    Conserved alveolate matches:  {n_conserved}")
    if has_compartment:
        top_comps = df["predicted_compartment"].value_counts().head(5)
        print(f"  Top predicted compartments:")
        for comp, n in top_comps.items():
            print(f"    {comp}: {n}")
    if has_reference:
        n_novel = (df["structural_novelty"] == "novel").sum()
        print(f"  Structurally novel:    {n_novel}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
