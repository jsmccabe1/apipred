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

__version__ = "1.1.0"

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
    """Load ESM-2 650M model via the fair-esm package."""
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def clean_sequence(seq):
    """Clean protein sequence: replace non-standard AAs, remove stops."""
    seq = seq.upper().replace("*", "")
    return "".join(c if c in VALID_AA else "X" for c in seq)


def _split_sequences(sequences, max_len, overlap):
    """Partition cleaned sequences into single-window and multi-window groups.

    Single-window proteins (len <= max_len) need no per-residue accumulator;
    they can be embedded and reduced to a mean vector inside one batch.

    Multi-window proteins (len > max_len) are split into overlapping windows
    here so each one can be processed as a self-contained group of windows
    (avoids holding per-residue accumulators for long-running batches).

    Returns:
        singles: list of (sid, length, seq), one per short protein
        multis: list of (sid, length, [(start, end, win_seq), ...])
        n_skipped: count of sequences shorter than 10 aa
    """
    singles = []
    multis = []
    n_skipped = 0
    stride = max_len - overlap

    for sid, _, seq in sequences:
        seq = clean_sequence(seq)
        if len(seq) < 10:
            n_skipped += 1
            continue
        if len(seq) <= max_len:
            singles.append((sid, len(seq), seq))
        else:
            wins = []
            pos = 0
            while pos < len(seq):
                end = min(pos + max_len, len(seq))
                wins.append((pos, end, seq[pos:end]))
                if end == len(seq):
                    break
                pos += stride
            multis.append((sid, len(seq), wins))

    return singles, multis, n_skipped


def _esm_forward(seqs, model, batch_converter, device, layers):
    """Run one batched ESM-2 forward pass; return CPU per-residue layer-mean tensor.

    Output shape: (batch, max_tokens, dim).
    """
    data = [(f"w{j}", s) for j, s in enumerate(seqs)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=list(layers), return_contacts=False)
    return torch.stack([out["representations"][l] for l in layers]).mean(dim=0).cpu()


def _embed_singles(singles, embeddings, model, batch_converter, device, layers,
                   batch_size, n_total_windows, n_done_ref, t0):
    """Embed all single-window proteins with adaptive batching.

    Each protein's embedding is computed and stored eagerly inside the batch
    that contains it, so no per-residue tensor outlives one forward pass.
    """
    # Length-sort so each batch has similar-length sequences (less padding waste).
    singles.sort(key=lambda x: x[1])
    cur_batch = batch_size
    i = 0
    while i < len(singles):
        batch = singles[i:i + cur_batch]
        seqs = [s for _, _, s in batch]
        try:
            mean_rep = _esm_forward(seqs, model, batch_converter, device, layers)
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "cuda":
                torch.cuda.empty_cache()
            if ("out of memory" in msg or "cuda" in msg) and cur_batch > 1:
                cur_batch = max(1, cur_batch // 2)
                continue
            # Single-sequence failure: drop and continue.
            sid = batch[0][0]
            embeddings.pop(sid, None)
            i += 1
            n_done_ref[0] += 1
            continue

        for j, (sid, win_len, _) in enumerate(batch):
            # Tokens [1, 1+win_len) are the residues; mean over residues for the protein vector.
            embeddings[sid] = mean_rep[j, 1:win_len + 1, :].mean(dim=0).numpy()

        i += len(batch)
        n_done_ref[0] += len(batch)
        elapsed = time.time() - t0
        rate = n_done_ref[0] / elapsed if elapsed > 0 else 0
        eta = (n_total_windows - n_done_ref[0]) / rate if rate > 0 else 0
        print(f"\r  [{n_done_ref[0]}/{n_total_windows} windows, batch={cur_batch}] "
              f"{rate:.1f} win/s ETA {eta:.0f}s", end="", flush=True)


def _embed_multis(multis, embeddings, model, batch_converter, device, layers,
                  n_total_windows, n_done_ref, t0):
    """Embed long proteins one at a time, finalising each before moving on.

    All windows of one protein go in a single batched forward pass, then
    overlap regions are averaged per residue, then the protein vector is the
    mean over all residues. Matches scripts/01_generate_embeddings.py exactly.
    The per-residue accumulator lives only for one protein at a time.
    """
    for sid, length, wins in multis:
        seqs = [w[2] for w in wins]
        try:
            mean_rep = _esm_forward(seqs, model, batch_converter, device, layers)
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "cuda":
                torch.cuda.empty_cache()
            # Long protein OOM: try one window at a time
            if "out of memory" in msg or "cuda" in msg:
                try:
                    chunks = []
                    for s in seqs:
                        chunks.append(_esm_forward([s], model, batch_converter,
                                                   device, layers))
                    # Pad chunks to common width and stack so the per-window loop below works.
                    max_t = max(c.shape[1] for c in chunks)
                    padded = torch.zeros(len(chunks), max_t, chunks[0].shape[2])
                    for k, c in enumerate(chunks):
                        padded[k, :c.shape[1], :] = c[0]
                    mean_rep = padded
                except RuntimeError:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    n_done_ref[0] += len(wins)
                    continue
            else:
                raise

        residue_sum = torch.zeros(length, mean_rep.shape[2])
        counts = torch.zeros(length)
        for j, (start, end, win_seq) in enumerate(wins):
            win_len = len(win_seq)
            residue_sum[start:end] += mean_rep[j, 1:win_len + 1, :]
            counts[start:end] += 1
        residue_emb = residue_sum / counts.unsqueeze(1).clamp(min=1)
        embeddings[sid] = residue_emb.mean(dim=0).numpy()

        n_done_ref[0] += len(wins)
        elapsed = time.time() - t0
        rate = n_done_ref[0] / elapsed if elapsed > 0 else 0
        eta = (n_total_windows - n_done_ref[0]) / rate if rate > 0 else 0
        print(f"\r  [{n_done_ref[0]}/{n_total_windows} windows, multi-window phase] "
              f"{rate:.1f} win/s ETA {eta:.0f}s", end="", flush=True)


def embed_proteome(sequences, model, alphabet, batch_converter,
                   device="cpu", batch_size=8, layers=(20, 24, 28, 33),
                   max_len=1022, overlap=200):
    """Embed all sequences with true batched ESM-2 inference.

    Two-phase: long proteins (>max_len aa) are processed first, one protein
    at a time, with all their windows in a single batched forward pass and
    immediate overlap-averaged finalisation. Then short proteins go through
    a length-sorted batched pipeline with eager per-protein mean reduction.
    Peak CPU memory is bounded by one batch's activations plus the final
    embeddings dict, regardless of proteome size.

    Per-residue overlap regions are averaged before the final residue mean,
    matching the training pipeline at
    Apicomplexa/scripts/01_generate_embeddings.py.
    """
    singles, multis, n_skipped = _split_sequences(sequences, max_len, overlap)
    n_singles = len(singles)
    n_multis = len(multis)
    n_total = n_singles + n_multis
    n_total_windows = n_singles + sum(len(w[2]) for w in multis)

    embeddings = {}
    n_done_ref = [0]
    t0 = time.time()

    if n_multis:
        _embed_multis(multis, embeddings, model, batch_converter, device, layers,
                      n_total_windows, n_done_ref, t0)
    if n_singles:
        _embed_singles(singles, embeddings, model, batch_converter, device, layers,
                       batch_size, n_total_windows, n_done_ref, t0)

    n_failed = n_skipped + (n_total - len(embeddings))
    elapsed = time.time() - t0
    print(f"\r  Embedded {len(embeddings)}/{n_total} proteins "
          f"in {elapsed:.1f}s ({n_failed} failed)         ")
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
    parser.add_argument("--version", action="version",
                        version=f"ApiPred {__version__}")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "models"

    essentiality_path = model_dir / "essentiality_ensemble.joblib"
    compartment_path = model_dir / "compartment_model.joblib"
    reference_path = model_dir / "reference_db.npz"

    has_essentiality = essentiality_path.exists()
    has_compartment = compartment_path.exists()
    has_reference = reference_path.exists()

    if not any([has_essentiality, has_compartment]):
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

        ensemble_preds = []
        ensemble_probs = []
        for fold_model in ess_data["ensemble"]:
            ensemble_preds.append(fold_model["regressor"].predict(X))
            ensemble_probs.append(fold_model["classifier"].predict_proba(X)[:, 1])

        predicted_scores = np.mean(ensemble_preds, axis=0)
        score_std = np.std(ensemble_preds, axis=0)
        essential_probs = np.mean(ensemble_probs, axis=0)
    else:
        predicted_scores = np.full(len(X), np.nan)
        score_std = np.full(len(X), np.nan)
        essential_probs = np.full(len(X), np.nan)

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
    if has_compartment:
        n_inv = (df["predicted_invasion"] == "yes").sum()
        n_specific = (df.get("invasion_specific") == "yes").sum()
        n_conserved = (df.get("invasion_specific") == "conserved_match").sum()
        print(f"  Predicted invasion (all):       {n_inv}")
        print(f"    Parasite-specific matches:    {n_specific}")
        print(f"    Conserved alveolate matches:  {n_conserved}")
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
