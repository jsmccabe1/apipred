#!/usr/bin/env python3
"""
ApiPred - Apicomplexan protein function predictor
====================================================

Predicts essentiality, subcellular compartment, invasion machinery membership,
and structural context for any apicomplexan proteome from ESM-2 embeddings.

For each protein, ApiPred predicts:
  - Essentiality (CRISPR fitness score + ensemble confidence)
  - Subcellular compartment (25-class hyperLOPIT classifier)
  - Invasion machinery membership
  - Structural novelty (distance from nearest characterised T. gondii protein)
  - Top 3 most similar characterised proteins
  - (optional) Per-window domain-level scoring
  - (optional) Rank against a pre-computed control proteome panel
  - (optional) Empirical invasion FDR against a background panel

Usage:
    python predict.py --input proteome.fasta --output predictions.tsv
    python predict.py --input proteome.fasta --output predictions.tsv --device cuda
    python predict.py --input proteome.fasta --output predictions.tsv \\
        --baseline-panel panel/ --per-window
"""

import argparse
import json
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

__version__ = "1.2.0"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
INVASION_COMPARTMENTS = {
    "rhoptries 1", "rhoptries 2", "micronemes", "dense granules",
    "IMC", "apical 1", "apical 2",
}

# Keywords identifying parasitism-specific invasion proteins (ROPKs, MICs, etc.)
# vs conserved alveolate proteins that happen to reside in invasion compartments.
# match_specificity is still useful as a cheap keyword signal when the top
# reference hit has an annotated name; it is NOT used to filter the invasion
# call itself (that job is now done by baseline-panel ranks + empirical FDR).
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


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDING PIPELINE (whole-protein)
# ══════════════════════════════════════════════════════════════════════════

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
            sid = batch[0][0]
            embeddings.pop(sid, None)
            i += 1
            n_done_ref[0] += 1
            continue

        for j, (sid, win_len, _) in enumerate(batch):
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
    """Embed long proteins one at a time, finalising each before moving on."""
    for sid, length, wins in multis:
        seqs = [w[2] for w in wins]
        try:
            mean_rep = _esm_forward(seqs, model, batch_converter, device, layers)
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "cuda":
                torch.cuda.empty_cache()
            if "out of memory" in msg or "cuda" in msg:
                try:
                    chunks = []
                    for s in seqs:
                        chunks.append(_esm_forward([s], model, batch_converter,
                                                   device, layers))
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


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDING PIPELINE (per-window, for domain-level scoring)
# ══════════════════════════════════════════════════════════════════════════

def embed_per_window(sequences, model, alphabet, batch_converter,
                     device="cpu", batch_size=8, window_size=200, stride=50,
                     layers=(20, 24, 28, 33)):
    """Split each protein into fixed-size windows and embed every window.

    Used for domain-level scoring: returns one embedding per window so the
    classifier can score each region of the protein independently. A 1500 aa
    protein with window_size=200, stride=50 yields ~27 windows.

    Short proteins (< window_size) still get one window covering the whole
    sequence, so every protein gets at least one window entry.

    Returns a list of dicts:
        [{"parent_id": sid, "window_idx": k, "start": s, "end": e,
          "embedding": np.ndarray(1280,)}, ...]
    """
    # Build window list: (parent_id, window_idx, start, end, window_seq)
    windows = []
    for sid, _, seq in sequences:
        seq = clean_sequence(seq)
        if len(seq) < 10:
            continue
        if len(seq) <= window_size:
            windows.append((sid, 0, 0, len(seq), seq))
            continue
        pos = 0
        idx = 0
        while pos < len(seq):
            end = min(pos + window_size, len(seq))
            windows.append((sid, idx, pos, end, seq[pos:end]))
            idx += 1
            if end == len(seq):
                break
            pos += stride

    if not windows:
        return []

    # Sort by length so batches have minimal padding
    windows.sort(key=lambda w: len(w[4]))
    n_total = len(windows)

    results = [None] * n_total
    # Remember insertion index per window so we can restore original order
    indexed = [(i, w) for i, w in enumerate(windows)]

    cur_batch = batch_size
    i = 0
    t0 = time.time()
    while i < len(indexed):
        batch = indexed[i:i + cur_batch]
        seqs = [w[4] for _, w in batch]
        try:
            mean_rep = _esm_forward(seqs, model, batch_converter, device, layers)
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "cuda":
                torch.cuda.empty_cache()
            if ("out of memory" in msg or "cuda" in msg) and cur_batch > 1:
                cur_batch = max(1, cur_batch // 2)
                continue
            i += 1
            continue

        for j, (orig_i, w) in enumerate(batch):
            sid, widx, start, end, win_seq = w
            emb = mean_rep[j, 1:len(win_seq) + 1, :].mean(dim=0).numpy()
            results[orig_i] = {
                "parent_id": sid,
                "window_idx": widx,
                "start": start,
                "end": end,
                "embedding": emb,
            }

        i += len(batch)
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n_total - i) / rate if rate > 0 else 0
        print(f"\r  [{i}/{n_total} per-window, batch={cur_batch}] "
              f"{rate:.1f} win/s ETA {eta:.0f}s", end="", flush=True)

    print()
    return [r for r in results if r is not None]


# ══════════════════════════════════════════════════════════════════════════
# FASTA PARSING
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING AND SCORING
# ══════════════════════════════════════════════════════════════════════════

def load_models(model_dir):
    """Load essentiality ensemble, compartment model, and reference database.

    Returns a dict with keys:
        essentiality: list of {regressor, classifier} fold models or None
        compartment: {"model": sklearn Pipeline, "classes": ndarray} or None
        reference: {"embeddings", "ids", "descriptions", "compartments",
                    "crispr_scores", "normed", "inv_mask", "noninv_mask"}
                    or None
    """
    import joblib
    model_dir = Path(model_dir)

    essentiality_path = model_dir / "essentiality_ensemble.joblib"
    compartment_path = model_dir / "compartment_model.joblib"
    reference_path = model_dir / "reference_db.npz"

    out = {"essentiality": None, "compartment": None, "reference": None}

    if essentiality_path.exists():
        ess_data = joblib.load(essentiality_path)
        out["essentiality"] = ess_data["ensemble"]

    if compartment_path.exists():
        comp_data = joblib.load(compartment_path)
        out["compartment"] = {
            "model": comp_data["model"],
            "classes": comp_data["classes"],
        }

    if reference_path.exists():
        ref_data = np.load(reference_path, allow_pickle=True)
        ref_embs = ref_data["embeddings"]
        ref_comps = ref_data["compartments"]
        ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
        ref_normed = ref_embs / (ref_norms + 1e-10)
        inv_mask = np.array([c in INVASION_COMPARTMENTS for c in ref_comps])
        out["reference"] = {
            "embeddings": ref_embs,
            "ids": ref_data["ids"],
            "descriptions": ref_data["descriptions"],
            "compartments": ref_comps,
            "crispr_scores": ref_data["crispr_scores"],
            "normed": ref_normed,
            "inv_mask": inv_mask,
            "noninv_mask": ~inv_mask,
        }

    return out


def score_embeddings(X, models):
    """Run essentiality + compartment + structural context on an embedding matrix.

    Returns a dict of numpy arrays, one entry per input row:
        predicted_crispr_score, score_std, essential_probability,
        predicted_compartment, compartment_confidence, invasion_probability,
        similar_1_id / similar_1_desc / similar_1_compartment / ...
        max_similarity_to_known, structural_novelty
    """
    n = len(X)
    results = {
        "predicted_crispr_score": np.full(n, np.nan),
        "score_std": np.full(n, np.nan),
        "essential_probability": np.full(n, np.nan),
        "predicted_compartment": np.full(n, "", dtype=object),
        "compartment_confidence": np.full(n, np.nan),
        "invasion_probability": np.full(n, np.nan),
    }

    if models["essentiality"] is not None:
        ensemble_preds = []
        ensemble_probs = []
        for fold in models["essentiality"]:
            ensemble_preds.append(fold["regressor"].predict(X))
            ensemble_probs.append(fold["classifier"].predict_proba(X)[:, 1])
        results["predicted_crispr_score"] = np.mean(ensemble_preds, axis=0)
        results["score_std"] = np.std(ensemble_preds, axis=0)
        results["essential_probability"] = np.mean(ensemble_probs, axis=0)

    if models["compartment"] is not None:
        comp = models["compartment"]
        probs = comp["model"].predict_proba(X)
        results["predicted_compartment"] = comp["classes"][np.argmax(probs, axis=1)]
        results["compartment_confidence"] = np.max(probs, axis=1)
        inv_mask = np.array([c in INVASION_COMPARTMENTS for c in comp["classes"]])
        results["invasion_probability"] = probs[:, inv_mask].sum(axis=1)

    if models["reference"] is not None:
        ref = models["reference"]
        X_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normed = X / (X_norms + 1e-10)
        sim = X_normed @ ref["normed"].T  # (n, n_ref)
        top3 = np.argsort(sim, axis=1)[:, -3:][:, ::-1]  # (n, 3) descending

        similar_ids = [[], [], []]
        similar_descs = [[], [], []]
        similar_comps = [[], [], []]
        similar_sims = [[], [], []]
        similar_crispr = [[], [], []]
        max_sim = np.zeros(n)
        novelty = np.full(n, "", dtype=object)

        for i in range(n):
            for rank in range(3):
                j = top3[i, rank]
                similar_ids[rank].append(str(ref["ids"][j]))
                similar_descs[rank].append(str(ref["descriptions"][j])[:50])
                similar_comps[rank].append(str(ref["compartments"][j]))
                similar_sims[rank].append(float(sim[i, j]))
                s = ref["crispr_scores"][j]
                similar_crispr[rank].append(float(s) if np.isfinite(s) else np.nan)
            max_sim[i] = float(sim[i, top3[i, 0]])
            novelty[i] = "novel" if max_sim[i] < 0.95 else "known_fold"

        for rank in range(3):
            k = rank + 1
            results[f"similar_{k}_id"] = np.array(similar_ids[rank], dtype=object)
            results[f"similar_{k}_desc"] = np.array(similar_descs[rank], dtype=object)
            results[f"similar_{k}_compartment"] = np.array(similar_comps[rank], dtype=object)
            results[f"similar_{k}_similarity"] = np.array(similar_sims[rank])
            results[f"similar_{k}_crispr"] = np.array(similar_crispr[rank])
        results["max_similarity_to_known"] = max_sim
        results["structural_novelty"] = novelty

    return results


# ══════════════════════════════════════════════════════════════════════════
# BASELINE PANEL RANKS (control proteome percentiles + empirical FDR)
# ══════════════════════════════════════════════════════════════════════════

def load_baseline_panel(panel_dir):
    """Load a pre-computed baseline panel.

    A panel directory is expected to contain panel.json with keys:
        apicomplexan: [list of organism names]
        background: [list of organism names]
        apicomplexan_invasion_probs: [sorted array of inv_prob values]
        background_invasion_probs: [sorted array of inv_prob values]
        background_fdr_table: [[score, fdr], ...] sorted by score ascending

    Returns the parsed panel dict, or None if panel_dir does not exist.
    """
    panel_dir = Path(panel_dir)
    panel_json = panel_dir / "panel.json"
    if not panel_json.exists():
        return None
    with open(panel_json) as f:
        panel = json.load(f)
    # Convert lists to sorted numpy arrays for fast lookup
    panel["apicomplexan_invasion_probs"] = np.asarray(
        panel["apicomplexan_invasion_probs"], dtype=np.float32)
    panel["background_invasion_probs"] = np.asarray(
        panel["background_invasion_probs"], dtype=np.float32)
    fdr_arr = np.asarray(panel["background_fdr_table"], dtype=np.float32)
    panel["_fdr_scores"] = fdr_arr[:, 0]
    panel["_fdr_values"] = fdr_arr[:, 1]
    return panel


def panel_ranks(inv_probs, panel):
    """Compute apicomplexan_rank, background_rank, invasion_fdr from a panel.

    Ranks are percentiles in the panel's invasion_probability distribution:
    100 = higher than everything in the panel, 0 = lower than everything.
    A real invasion candidate should be in the top few percent of the
    apicomplexan distribution AND close to 100 against the background.

    invasion_fdr is the empirical false discovery rate from the background
    panel: fraction of background proteins that score at least as high.
    """
    apico = panel["apicomplexan_invasion_probs"]
    bg = panel["background_invasion_probs"]
    fdr_scores = panel["_fdr_scores"]
    fdr_values = panel["_fdr_values"]

    # searchsorted gives the number of panel entries <= each query value.
    apico_rank = 100.0 * np.searchsorted(apico, inv_probs, side="right") / len(apico)
    bg_rank = 100.0 * np.searchsorted(bg, inv_probs, side="right") / len(bg)
    # FDR: for each query, find the highest FDR-score threshold <= the query
    # and return the corresponding FDR value. Queries above all thresholds get 0.
    idx = np.searchsorted(fdr_scores, inv_probs, side="right") - 1
    idx = np.clip(idx, 0, len(fdr_values) - 1)
    fdr = fdr_values[idx]
    # Queries below the smallest threshold: FDR is undefined; use 1.0
    fdr = np.where(inv_probs < fdr_scores[0], 1.0, fdr)
    return apico_rank, bg_rank, fdr


# ══════════════════════════════════════════════════════════════════════════
# PER-WINDOW AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

def aggregate_per_window(window_rows, window_scores, models):
    """Aggregate per-window scoring results to per-protein.

    window_rows: list of dicts from embed_per_window
    window_scores: dict of arrays from score_embeddings over all windows

    Returns per-parent dict:
        {sid: {"best_window_score": ..., "best_window_start": ...,
               "best_window_end": ..., "best_window_match": ...,
               "n_invasion_windows": ...}}
    """
    per_parent = {}
    for i, w in enumerate(window_rows):
        sid = w["parent_id"]
        inv_p = float(window_scores["invasion_probability"][i])
        comp = str(window_scores["predicted_compartment"][i])
        sim_1 = window_scores.get("similar_1_desc", [""] * len(window_rows))[i]
        sim_1_id = window_scores.get("similar_1_id", [""] * len(window_rows))[i]

        entry = per_parent.setdefault(sid, {
            "best_window_score": -1.0,
            "best_window_start": -1,
            "best_window_end": -1,
            "best_window_match": "",
            "best_window_match_id": "",
            "n_invasion_windows": 0,
        })
        if comp in INVASION_COMPARTMENTS:
            entry["n_invasion_windows"] += 1
        if inv_p > entry["best_window_score"]:
            entry["best_window_score"] = inv_p
            entry["best_window_start"] = w["start"]
            entry["best_window_end"] = w["end"]
            entry["best_window_match"] = str(sim_1)
            entry["best_window_match_id"] = str(sim_1_id)
    return per_parent


# ══════════════════════════════════════════════════════════════════════════
# ROW BUILDING
# ══════════════════════════════════════════════════════════════════════════

def build_rows(seq_ids, desc_map, len_map, scores, invasion_threshold=0.5,
               panel_out=None, per_window_out=None, fdr_threshold=None):
    """Assemble the final output dataframe.

    scores: output of score_embeddings()
    panel_out: optional (apico_rank, bg_rank, fdr) tuple from panel_ranks()
    per_window_out: optional {sid: {...}} from aggregate_per_window()
    fdr_threshold: if given and a panel is provided, predicted_invasion
                   requires invasion_fdr <= fdr_threshold AS WELL AS
                   invasion_probability > invasion_threshold

    Returns a pandas DataFrame with the full output schema.
    """
    n = len(seq_ids)
    rows = []
    apico_rank = bg_rank = fdr = None
    if panel_out is not None:
        apico_rank, bg_rank, fdr = panel_out

    for i, sid in enumerate(seq_ids):
        score = float(scores["predicted_crispr_score"][i])
        std = float(scores["score_std"][i])
        ess_prob = float(scores["essential_probability"][i])
        inv_prob = float(scores["invasion_probability"][i])

        if np.isnan(score):
            ess_class = "no_model"
        elif score < -3:
            ess_class = "essential"
        elif score < -1:
            ess_class = "important"
        else:
            ess_class = "dispensable"

        if not np.isnan(std):
            if std < 0.5:
                confidence = "high"
            elif std < 1.0:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = ""

        # predicted_invasion: inv_prob > threshold, and optionally FDR-constrained
        if np.isnan(inv_prob):
            invasion_call = ""
        elif inv_prob > invasion_threshold:
            if fdr is not None and fdr_threshold is not None:
                invasion_call = "yes" if fdr[i] <= fdr_threshold else "no"
            else:
                invasion_call = "yes"
        else:
            invasion_call = "no"

        row = {
            "protein_id": sid,
            "description": desc_map.get(sid, "")[:80],
            "length": len_map.get(sid, 0),
            "predicted_crispr_score": round(score, 3) if not np.isnan(score) else "",
            "score_std": round(std, 3) if not np.isnan(std) else "",
            "essential_probability": round(ess_prob, 3) if not np.isnan(ess_prob) else "",
            "essentiality_class": ess_class,
            "essentiality_confidence": confidence,
            "predicted_compartment": str(scores["predicted_compartment"][i]),
            "compartment_confidence": (
                round(float(scores["compartment_confidence"][i]), 3)
                if not np.isnan(scores["compartment_confidence"][i]) else ""),
            "invasion_probability": round(inv_prob, 3) if not np.isnan(inv_prob) else "",
            "predicted_invasion": invasion_call,
        }

        if panel_out is not None:
            row["apicomplexan_rank"] = round(float(apico_rank[i]), 2)
            row["background_rank"] = round(float(bg_rank[i]), 2)
            row["invasion_fdr"] = round(float(fdr[i]), 4)

        if per_window_out is not None:
            w = per_window_out.get(sid)
            if w is not None:
                row["best_window_score"] = round(float(w["best_window_score"]), 3)
                row["best_window_start"] = int(w["best_window_start"])
                row["best_window_end"] = int(w["best_window_end"])
                row["best_window_match"] = w["best_window_match"]
                row["best_window_match_id"] = w["best_window_match_id"]
                row["n_invasion_windows"] = int(w["n_invasion_windows"])
            else:
                row["best_window_score"] = ""
                row["best_window_start"] = ""
                row["best_window_end"] = ""
                row["best_window_match"] = ""
                row["best_window_match_id"] = ""
                row["n_invasion_windows"] = 0

        # Structural context columns
        if "similar_1_id" in scores:
            for k in (1, 2, 3):
                row[f"similar_{k}_id"] = str(scores[f"similar_{k}_id"][i])
                row[f"similar_{k}_desc"] = str(scores[f"similar_{k}_desc"][i])
                row[f"similar_{k}_compartment"] = str(scores[f"similar_{k}_compartment"][i])
                row[f"similar_{k}_similarity"] = round(float(scores[f"similar_{k}_similarity"][i]), 4)
                c = scores[f"similar_{k}_crispr"][i]
                row[f"similar_{k}_crispr"] = round(float(c), 2) if np.isfinite(c) else ""
            row["max_similarity_to_known"] = round(float(scores["max_similarity_to_known"][i]), 4)
            row["structural_novelty"] = str(scores["structural_novelty"][i])
            # Keyword classifier on top hit (fast sanity signal, not a filter)
            row["match_specificity"] = classify_match_specificity(
                scores["similar_1_desc"][i])

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# STRUCTURAL VALIDATION STUB (ESMFold + Foldseek)
# ══════════════════════════════════════════════════════════════════════════

def validate_structures_stub(top_n, top_df, reference_models):
    """Placeholder for ESMFold + Foldseek validation.

    ESMFold requires ~24 GB GPU memory in default mode, which exceeds what
    most workstation GPUs provide. Rather than silently producing nonsense,
    this function prints the install recipe and exits the validation step.

    When enabled, the intended flow is:
      1. Run ESMFold on the top_n candidates and on each candidate's top
         T. gondii reference protein.
      2. Compute pLDDT per residue (quality), TM-score via TMalign/Foldseek
         between candidate and reference.
      3. Return a table with (pLDDT_mean, tm_score_to_top_match,
         structurally_validated) for each candidate, where
         structurally_validated = True iff TM-score >= 0.5.
    """
    print()
    print("=" * 60)
    print("Structural validation stub (not executed)")
    print("=" * 60)
    print(f"  Requested: top {top_n} candidates")
    print(f"  Requires:  ESMFold (>=24 GB GPU) + Foldseek")
    print()
    print("  Install instructions:")
    print("    pip install fair-esm[esmfold]")
    print("    # Foldseek: https://github.com/steineggerlab/foldseek")
    print()
    print("  Once installed, rerun with:")
    print("    --validate-structures N --esmfold-device cuda:0")
    print("=" * 60)
    return None


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

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
    parser.add_argument("--baseline-panel", default=None,
                        help="Path to a precomputed control proteome panel "
                             "directory. Adds apicomplexan_rank, "
                             "background_rank, and invasion_fdr columns.")
    parser.add_argument("--fdr-threshold", type=float, default=None,
                        help="If set with --baseline-panel, predicted_invasion "
                             "requires invasion_fdr <= this value in addition "
                             "to invasion_probability > 0.5.")
    parser.add_argument("--per-window", action="store_true",
                        help="Also run per-window domain-level scoring. "
                             "Emits best_window_* columns in the main output "
                             "plus a sidecar TSV with all windows.")
    parser.add_argument("--window-size", type=int, default=200,
                        help="Per-window mode window length (default: 200)")
    parser.add_argument("--window-stride", type=int, default=50,
                        help="Per-window mode stride (default: 50)")
    parser.add_argument("--validate-structures", type=int, default=0,
                        metavar="N",
                        help="Opt-in: run ESMFold + Foldseek on top N "
                             "candidates. Requires ~24 GB GPU; stubbed with "
                             "install recipe if unavailable.")
    parser.add_argument("--version", action="version",
                        version=f"ApiPred {__version__}")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "models"

    # ─── Load models ───
    print(f"Loading pre-trained models from {model_dir}/")
    models = load_models(model_dir)
    if models["essentiality"] is None and models["compartment"] is None:
        print(f"  No models found. Proceeding with embedding + structural context only.\n")

    # ─── Baseline panel ───
    panel = None
    if args.baseline_panel:
        print(f"Loading baseline panel from {args.baseline_panel}/")
        panel = load_baseline_panel(args.baseline_panel)
        if panel is None:
            print(f"  WARNING: panel.json not found in {args.baseline_panel}; "
                  f"skipping rank columns")
        else:
            print(f"  Apicomplexan panel: {len(panel['apicomplexan_invasion_probs'])} proteins, "
                  f"background panel: {len(panel['background_invasion_probs'])} proteins")

    # ─── Parse input ───
    print(f"Reading {args.input}...")
    sequences = parse_fasta(args.input)
    print(f"  {len(sequences)} proteins")
    if len(sequences) == 0:
        print("ERROR: No valid sequences found")
        sys.exit(1)

    # ─── Embed (whole-protein) ───
    print("Loading ESM-2...")
    model, alphabet, batch_converter = load_esm2_model()
    model = model.to(args.device)

    print(f"Embedding {len(sequences)} proteins (batch_size={args.batch_size})...")
    embeddings = embed_proteome(sequences, model, alphabet, batch_converter,
                                device=args.device, batch_size=args.batch_size)

    seq_ids = [sid for sid, _, _ in sequences if sid in embeddings]
    X = np.array([embeddings[sid] for sid in seq_ids])
    desc_map = {sid: desc for sid, desc, _ in sequences}
    len_map = {sid: len(seq) for sid, _, seq in sequences}

    # ─── Per-window embedding (optional) ───
    per_window_out = None
    if args.per_window:
        print(f"Per-window scoring (window={args.window_size}, "
              f"stride={args.window_stride})...")
        # Only per-window score the proteins whose whole-protein embedding
        # succeeded, to keep counts consistent
        surviving = [(sid, desc, seq) for sid, desc, seq in sequences
                     if sid in embeddings]
        window_rows = embed_per_window(
            surviving, model, alphabet, batch_converter,
            device=args.device, batch_size=args.batch_size,
            window_size=args.window_size, stride=args.window_stride)

    # Free GPU memory now that embedding is done
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ─── Score whole-protein embeddings ───
    print("Scoring whole-protein embeddings...")
    scores = score_embeddings(X, models)

    # ─── Score per-window embeddings ───
    if args.per_window:
        print(f"Scoring {len(window_rows)} windows...")
        Xw = np.array([w["embedding"] for w in window_rows])
        window_scores = score_embeddings(Xw, models)
        per_window_out = aggregate_per_window(window_rows, window_scores, models)

        # Sidecar per-window TSV
        sidecar_path = Path(args.output).with_suffix("")
        sidecar_path = Path(str(sidecar_path) + ".windows.tsv")
        print(f"Writing per-window sidecar: {sidecar_path}")
        sidecar_rows = []
        for i, w in enumerate(window_rows):
            sidecar_rows.append({
                "parent_id": w["parent_id"],
                "window_idx": w["window_idx"],
                "start": w["start"],
                "end": w["end"],
                "predicted_compartment": str(window_scores["predicted_compartment"][i]),
                "compartment_confidence": round(float(window_scores["compartment_confidence"][i]), 3),
                "invasion_probability": round(float(window_scores["invasion_probability"][i]), 3),
                "top_match_id": str(window_scores.get("similar_1_id", [""] * len(window_rows))[i]),
                "top_match_desc": str(window_scores.get("similar_1_desc", [""] * len(window_rows))[i]),
                "top_match_similarity": round(float(window_scores.get("similar_1_similarity", [0] * len(window_rows))[i]), 4),
            })
        pd.DataFrame(sidecar_rows).to_csv(sidecar_path, sep="\t", index=False)

    # ─── Panel ranks ───
    panel_out = None
    if panel is not None:
        panel_out = panel_ranks(scores["invasion_probability"], panel)

    # ─── Build rows ───
    print("Generating predictions...")
    df = build_rows(seq_ids, desc_map, len_map, scores,
                    panel_out=panel_out, per_window_out=per_window_out,
                    fdr_threshold=args.fdr_threshold)
    if models["essentiality"] is not None:
        df = df.sort_values("predicted_crispr_score")
    df.to_csv(args.output, sep="\t", index=False)

    # ─── Structural validation (opt-in stub) ───
    if args.validate_structures:
        validate_structures_stub(args.validate_structures, df, models["reference"])

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"ApiPred results: {args.output}")
    print(f"{'='*60}")
    print(f"  Proteins analysed:     {len(df)}")
    if models["essentiality"] is not None:
        n_ess = (df["essentiality_class"] == "essential").sum()
        n_imp = (df["essentiality_class"] == "important").sum()
        n_hi = (df["essentiality_confidence"] == "high").sum()
        print(f"  Predicted essential:   {n_ess}")
        print(f"  Predicted important:   {n_imp}")
        print(f"  High-confidence calls: {n_hi}")
    if models["compartment"] is not None:
        n_inv = (df["predicted_invasion"] == "yes").sum()
        print(f"  Predicted invasion:    {n_inv}")
        if panel is not None:
            n_top1 = (df["apicomplexan_rank"] >= 99).sum()
            n_bg_low = (df["background_rank"] >= 95).sum()
            n_fdr = (df["invasion_fdr"] <= 0.05).sum()
            print(f"  apicomplexan_rank>=99: {n_top1}")
            print(f"  background_rank>=95:   {n_bg_low}")
            print(f"  invasion_fdr<=0.05:    {n_fdr}")
        top_comps = df["predicted_compartment"].value_counts().head(5)
        print(f"  Top predicted compartments:")
        for comp, n in top_comps.items():
            print(f"    {comp}: {n}")
    if models["reference"] is not None:
        n_novel = (df["structural_novelty"] == "novel").sum()
        print(f"  Structurally novel:    {n_novel}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
