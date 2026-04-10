"""
Smoke tests for predict.py.

Runs predict.py end-to-end on the bundled T. gondii example and verifies
the output schema, a few sanity invariants, and the optional per-window
and baseline-panel modes when applicable.

These tests download ESM-2 (~2.5 GB) the first time they run, so they are
slow on a fresh CI runner. They exercise the full pipeline (parse, embed,
classify, structural lookup, write).
"""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent

# Base schema (no panel, no per-window). Keep this list in sync with
# build_rows in predict.py.
EXPECTED_BASE_COLUMNS = [
    "protein_id",
    "description",
    "length",
    "predicted_crispr_score",
    "score_std",
    "essential_probability",
    "essentiality_class",
    "essentiality_confidence",
    "predicted_compartment",
    "compartment_confidence",
    "invasion_probability",
    "predicted_invasion",
    "similar_1_id",
    "similar_1_desc",
    "similar_1_compartment",
    "similar_1_similarity",
    "similar_1_crispr",
    "similar_2_id",
    "similar_2_desc",
    "similar_2_compartment",
    "similar_2_similarity",
    "similar_2_crispr",
    "similar_3_id",
    "similar_3_desc",
    "similar_3_compartment",
    "similar_3_similarity",
    "similar_3_crispr",
    "max_similarity_to_known",
    "structural_novelty",
    "match_specificity",
]

PER_WINDOW_COLUMNS = [
    "best_window_score",
    "best_window_start",
    "best_window_end",
    "best_window_match",
    "best_window_match_id",
    "n_invasion_windows",
]

PANEL_COLUMNS = [
    "apicomplexan_rank",
    "background_rank",
    "invasion_fdr",
]


def _run_predict(out_path, extra_args=None):
    cmd = [sys.executable, str(REPO / "predict.py"),
           "--input", str(REPO / "examples/test_tg.fasta"),
           "--output", str(out_path),
           "--device", "cpu",
           "--batch-size", "2"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def _assert_ok(result, out_path):
    assert result.returncode == 0, (
        f"predict.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert out_path.exists(), "predict.py did not write the output file"


def test_smoke_base(tmp_path):
    """Base run: no panel, no per-window. Must match the canonical schema."""
    out = tmp_path / "predictions.tsv"
    result = _run_predict(out)
    _assert_ok(result, out)

    df = pd.read_csv(out, sep="\t")
    assert len(df) == 10

    assert list(df.columns) == EXPECTED_BASE_COLUMNS, (
        "output schema drifted: "
        f"missing {set(EXPECTED_BASE_COLUMNS) - set(df.columns)}, "
        f"extra {set(df.columns) - set(EXPECTED_BASE_COLUMNS)}")

    # Self-match sanity: 9/10 example proteins are in the reference DB
    n_self_match = (df["max_similarity_to_known"] >= 0.99).sum()
    assert n_self_match >= 9

    n_invasion = (df["predicted_invasion"] == "yes").sum()
    assert n_invasion >= 7

    assert df["essential_probability"].between(0, 1).all()
    assert df["invasion_probability"].between(0, 1).all()
    assert df["compartment_confidence"].between(0, 1).all()


def test_smoke_per_window(tmp_path):
    """Per-window mode: adds best_window_* columns and a sidecar TSV."""
    out = tmp_path / "predictions.tsv"
    result = _run_predict(out, ["--per-window",
                                "--window-size", "200",
                                "--window-stride", "100"])
    _assert_ok(result, out)

    df = pd.read_csv(out, sep="\t")
    for col in PER_WINDOW_COLUMNS:
        assert col in df.columns, f"per-window mode missing column {col}"

    sidecar = out.with_suffix("").with_suffix(".windows.tsv")
    assert sidecar.exists(), "per-window sidecar TSV not written"
    sidecar_df = pd.read_csv(sidecar, sep="\t")
    assert len(sidecar_df) > 10, (
        f"expected more than 10 windows for the 10 example proteins, "
        f"got {len(sidecar_df)}")
    assert {"parent_id", "window_idx", "start", "end",
            "invasion_probability"}.issubset(sidecar_df.columns)


def _make_fake_panel(tmp_path):
    """Build a minimal panel.json with synthetic distributions for testing."""
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    panel = {
        "version": "1.0-test",
        "organisms": [],
        "apicomplexan_invasion_probs": [round(i / 100.0, 4) for i in range(101)],
        "background_invasion_probs": [round((i / 100.0) ** 2, 4) for i in range(101)],
        # Dense FDR grid from 0 to 1 in 0.01 steps
        "background_fdr_table": [[round(i / 100.0, 4),
                                  round(max(0.0, 1.0 - (i / 100.0) ** 0.5), 4)]
                                 for i in range(101)],
    }
    with open(panel_dir / "panel.json", "w") as f:
        json.dump(panel, f)
    return panel_dir


def test_smoke_with_panel(tmp_path):
    """--baseline-panel adds apicomplexan_rank, background_rank, invasion_fdr."""
    panel_dir = _make_fake_panel(tmp_path)
    out = tmp_path / "predictions.tsv"
    result = _run_predict(out, ["--baseline-panel", str(panel_dir)])
    _assert_ok(result, out)

    df = pd.read_csv(out, sep="\t")
    for col in PANEL_COLUMNS:
        assert col in df.columns, f"panel mode missing column {col}"
    assert df["apicomplexan_rank"].between(0, 100).all()
    assert df["background_rank"].between(0, 100).all()
    assert df["invasion_fdr"].between(0, 1).all()


def test_extract_subcommand(tmp_path):
    """extract.py end-to-end: filter + dedup + FASTA extract."""
    # First produce a small predictions file
    out = tmp_path / "predictions.tsv"
    result = _run_predict(out)
    _assert_ok(result, out)

    cand_dir = tmp_path / "candidates"
    result = subprocess.run(
        [sys.executable, str(REPO / "extract.py"),
         str(out),
         "--invasion-only",
         "--dedup", "none",
         "--source-fasta", str(REPO / "examples/test_tg.fasta"),
         "--no-html",
         "--output", str(cand_dir)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"extract.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    tsv = cand_dir / "candidates.tsv"
    fasta = cand_dir / "candidates.fasta"
    assert tsv.exists()
    assert fasta.exists()

    cand_df = pd.read_csv(tsv, sep="\t")
    assert (cand_df["predicted_invasion"] == "yes").all()
    # At least some candidates should have been written to FASTA
    n_fasta = sum(1 for line in open(fasta) if line.startswith(">"))
    assert n_fasta == len(cand_df)
