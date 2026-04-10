#!/usr/bin/env python3
"""
apipred extract - filter, dedupe, and export ApiPred predictions.

Turns a raw ApiPred predictions TSV into an actionable candidate set ready
for downstream analysis (phylogeny, structure prediction, decontamination
checks). Replaces ad-hoc awk and pandas one-liners with a consistent pipeline:

  1. Filter by compartment, match_specificity, ranks, FDR, similarity, novelty
  2. Dedup redundant ORFs (TransDecoder style) to gene level
  3. Optionally annotate with contamination calls from a user Diamond/BLAST TSV
  4. Emit a candidate FASTA (requires --source-fasta), a trimmed metadata TSV,
     and an HTML evidence-card report

Usage:
    python extract.py predictions.tsv \\
        --min-invasion-prob 0.5 \\
        --min-apicomplexan-rank 99 \\
        --max-invasion-fdr 0.05 \\
        --dedup transdecoder \\
        --top 50 \\
        --source-fasta proteome.fasta \\
        --output candidates/
"""

import argparse
import re
import sys
import html
from pathlib import Path
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Dedup strategies
# ──────────────────────────────────────────────────────────────────

TRANSDECODER_ORF_RE = re.compile(r"\.p\d+$")
TRANSDECODER_ISOFORM_RE = re.compile(r"_i\d+$")


def dedup_transdecoder_gene(df):
    """Collapse TransDecoder ORF IDs to the gene level.

    TransDecoder protein IDs look like:
        TRINITY_DN123_c0_g1_i1.p1
        TRINITY_DN123_c0_g1_i1.p2
        TRINITY_DN123_c0_g1_i2.p1
    All three are the same gene (g1); different isoforms/ORFs. This keeps
    the single best row per gene (best = highest invasion_probability, or
    highest apicomplexan_rank if available).

    Proteins whose IDs don't match the TransDecoder pattern are kept as-is.
    """
    def gene_key(pid):
        pid = str(pid)
        pid = TRANSDECODER_ORF_RE.sub("", pid)
        pid = TRANSDECODER_ISOFORM_RE.sub("", pid)
        return pid

    df = df.copy()
    df["_gene_key"] = df["protein_id"].apply(gene_key)
    sort_key = "apicomplexan_rank" if "apicomplexan_rank" in df.columns else "invasion_probability"
    df = df.sort_values(sort_key, ascending=False)
    df = df.drop_duplicates(subset="_gene_key", keep="first")
    df = df.drop(columns="_gene_key")
    return df


def dedup_identical_top_hit(df):
    """Collapse rows that share the same top T. gondii match.

    Useful when multiple paralogs all hit the same reference protein and
    you only want one representative per reference hit.
    """
    if "similar_1_id" not in df.columns:
        return df
    sort_key = "apicomplexan_rank" if "apicomplexan_rank" in df.columns else "invasion_probability"
    return df.sort_values(sort_key, ascending=False).drop_duplicates(
        subset="similar_1_id", keep="first")


DEDUP_STRATEGIES = {
    "transdecoder": dedup_transdecoder_gene,
    "top-hit": dedup_identical_top_hit,
    "none": lambda df: df,
}


# ──────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────

INVASION_COMPARTMENTS = {
    "rhoptries 1", "rhoptries 2", "micronemes", "dense granules",
    "IMC", "apical 1", "apical 2",
}


def apply_filters(df, args):
    """Apply command-line filters to the predictions dataframe."""
    n0 = len(df)

    if args.invasion_only:
        df = df[df["predicted_invasion"] == "yes"]

    if args.compartment:
        df = df[df["predicted_compartment"].isin(args.compartment)]

    if args.invasion_compartment_only:
        df = df[df["predicted_compartment"].isin(INVASION_COMPARTMENTS)]

    if args.match_specificity:
        if "match_specificity" in df.columns:
            df = df[df["match_specificity"].isin(args.match_specificity)]

    if args.min_invasion_prob is not None:
        df = df[df["invasion_probability"].astype(float) >= args.min_invasion_prob]

    if args.min_apicomplexan_rank is not None:
        if "apicomplexan_rank" in df.columns:
            df = df[df["apicomplexan_rank"].astype(float) >= args.min_apicomplexan_rank]
        else:
            print(f"  WARNING: --min-apicomplexan-rank requires a panel-annotated "
                  f"predictions file; column not found, skipping this filter.")

    if args.min_background_rank is not None:
        if "background_rank" in df.columns:
            df = df[df["background_rank"].astype(float) >= args.min_background_rank]
        else:
            print(f"  WARNING: --min-background-rank requires a panel-annotated "
                  f"predictions file; column not found, skipping this filter.")

    if args.max_invasion_fdr is not None:
        if "invasion_fdr" in df.columns:
            df = df[df["invasion_fdr"].astype(float) <= args.max_invasion_fdr]
        else:
            print(f"  WARNING: --max-invasion-fdr requires a panel-annotated "
                  f"predictions file; column not found, skipping this filter.")

    if args.min_similarity is not None:
        if "max_similarity_to_known" in df.columns:
            df = df[df["max_similarity_to_known"].astype(float) >= args.min_similarity]

    if args.exclude_novel and "structural_novelty" in df.columns:
        df = df[df["structural_novelty"] == "known_fold"]

    if args.only_novel and "structural_novelty" in df.columns:
        df = df[df["structural_novelty"] == "novel"]

    if args.min_length is not None:
        df = df[df["length"].astype(int) >= args.min_length]

    print(f"  Filters: {n0} -> {len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────────
# Contamination annotation
# ──────────────────────────────────────────────────────────────────

def load_taxonomy_calls(path):
    """Load a user-supplied taxonomy TSV mapping protein_id -> broad_taxon.

    Expected format: tab-separated, two columns, header optional:
        protein_id    taxonomy
        TRINITY_XXX   apicomplexa
        TRINITY_YYY   ciliate

    Users can generate this from Diamond blastp output + a taxonomy lookup,
    or from Kraken2 classification of the underlying transcripts. ApiPred
    does not run Diamond itself.
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["protein_id", "taxonomy"])
    # If the first row looks like a header, drop it
    if df.iloc[0]["protein_id"].lower() == "protein_id":
        df = df.iloc[1:].reset_index(drop=True)
    return dict(zip(df["protein_id"], df["taxonomy"]))


def annotate_taxonomy(df, tax_map):
    """Add a lca_taxonomy column from the user-supplied map."""
    df = df.copy()
    df["lca_taxonomy"] = df["protein_id"].map(tax_map).fillna("unclassified")
    return df


# ──────────────────────────────────────────────────────────────────
# FASTA extraction
# ──────────────────────────────────────────────────────────────────

def parse_fasta_ids(fasta_path, wanted_ids):
    """Stream a FASTA file and return {id: sequence} for ids in wanted_ids."""
    wanted = set(wanted_ids)
    found = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id in wanted and current_seq:
                    found[current_id] = "".join(current_seq)
                current_id = line[1:].split(None, 1)[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id in wanted and current_seq:
            found[current_id] = "".join(current_seq)
    return found


def write_fasta(out_path, id_to_seq, desc_map=None):
    """Write a FASTA with one sequence per line wrapped at 60 chars."""
    with open(out_path, "w") as f:
        for sid, seq in id_to_seq.items():
            desc = (desc_map or {}).get(sid, "")
            header = f">{sid} {desc}".rstrip()
            f.write(header + "\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


# ──────────────────────────────────────────────────────────────────
# HTML evidence cards
# ──────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ApiPred candidates</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.4em; }}
.meta {{ color: #666; font-size: 0.9em; margin-bottom: 2em; }}
.card {{ border: 1px solid #ddd; border-radius: 6px; padding: 1em; margin-bottom: 1em;
        background: #fafafa; }}
.card h2 {{ font-size: 1.1em; margin: 0 0 0.5em 0; font-family: monospace; }}
.pair {{ display: inline-block; margin-right: 1.5em; font-size: 0.9em; }}
.pair b {{ color: #555; }}
table {{ border-collapse: collapse; font-size: 0.85em; margin-top: 0.5em; }}
table th, table td {{ padding: 3px 8px; text-align: left; border-bottom: 1px solid #eee; }}
table th {{ background: #f0f0f0; }}
.fam-AMA {{ color: #c33; }}
.fam-RON {{ color: #c73; }}
.fam-MIC {{ color: #390; }}
.fam-ROP {{ color: #639; }}
.fam-GRA {{ color: #369; }}
</style></head><body>
<h1>ApiPred candidates</h1>
<div class="meta">Generated from {source} | {n} candidates</div>
{cards}
</body></html>
"""

CARD_TEMPLATE = """<div class="card">
<h2>{protein_id}</h2>
<div class="pair"><b>length</b> {length} aa</div>
<div class="pair"><b>invasion_prob</b> {invasion_probability}</div>
{rank_pair}
{fdr_pair}
<div class="pair"><b>compartment</b> {predicted_compartment} ({compartment_confidence})</div>
<div class="pair"><b>essentiality</b> {essentiality_class} ({predicted_crispr_score})</div>
<br>
<table>
<tr><th>rank</th><th>T. gondii hit</th><th>description</th><th>compartment</th><th>similarity</th></tr>
{similar_rows}
</table>
</div>
"""


def build_card(row):
    rank_pair = ""
    fdr_pair = ""
    if "apicomplexan_rank" in row.index:
        rank_pair = f'<div class="pair"><b>apico_rank</b> {row["apicomplexan_rank"]}%ile</div>'
        rank_pair += f'<div class="pair"><b>bg_rank</b> {row["background_rank"]}%ile</div>'
    if "invasion_fdr" in row.index:
        fdr_pair = f'<div class="pair"><b>invasion_fdr</b> {row["invasion_fdr"]}</div>'

    similar_rows = ""
    for k in (1, 2, 3):
        if f"similar_{k}_id" in row.index:
            similar_rows += (
                f"<tr><td>{k}</td>"
                f"<td><code>{html.escape(str(row[f'similar_{k}_id']))}</code></td>"
                f"<td>{html.escape(str(row[f'similar_{k}_desc']))}</td>"
                f"<td>{html.escape(str(row[f'similar_{k}_compartment']))}</td>"
                f"<td>{row[f'similar_{k}_similarity']}</td></tr>"
            )

    return CARD_TEMPLATE.format(
        protein_id=html.escape(str(row["protein_id"])),
        length=row["length"],
        invasion_probability=row["invasion_probability"],
        rank_pair=rank_pair,
        fdr_pair=fdr_pair,
        predicted_compartment=html.escape(str(row["predicted_compartment"])),
        compartment_confidence=row["compartment_confidence"],
        essentiality_class=row["essentiality_class"],
        predicted_crispr_score=row["predicted_crispr_score"],
        similar_rows=similar_rows,
    )


def write_html_report(out_path, df, source_name):
    cards = "\n".join(build_card(row) for _, row in df.iterrows())
    with open(out_path, "w") as f:
        f.write(HTML_TEMPLATE.format(
            source=html.escape(source_name), n=len(df), cards=cards))


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter, dedupe, and export ApiPred predictions into "
                    "an actionable candidate set.")
    parser.add_argument("predictions", help="Input ApiPred predictions TSV")
    parser.add_argument("--output", "-o", default="candidates",
                        help="Output directory (default: candidates/)")
    parser.add_argument("--source-fasta", default=None,
                        help="Original FASTA to extract candidate sequences from")

    # Filters
    parser.add_argument("--invasion-only", action="store_true",
                        help="Keep only predicted_invasion=yes rows")
    parser.add_argument("--invasion-compartment-only", action="store_true",
                        help="Keep only rows predicted into an invasion compartment")
    parser.add_argument("--compartment", nargs="+", default=None,
                        help="Keep only these predicted compartments")
    parser.add_argument("--match-specificity", nargs="+", default=None,
                        choices=["parasite_specific", "conserved", "unknown", "unclassified"],
                        help="Keep only these match_specificity values")
    parser.add_argument("--min-invasion-prob", type=float, default=None)
    parser.add_argument("--min-apicomplexan-rank", type=float, default=None,
                        help="Minimum apicomplexan_rank (0-100, requires panel)")
    parser.add_argument("--min-background-rank", type=float, default=None,
                        help="Minimum background_rank (0-100, requires panel)")
    parser.add_argument("--max-invasion-fdr", type=float, default=None,
                        help="Maximum invasion_fdr (requires panel)")
    parser.add_argument("--min-similarity", type=float, default=None,
                        help="Minimum max_similarity_to_known")
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--exclude-novel", action="store_true",
                        help="Drop structural_novelty=novel rows")
    parser.add_argument("--only-novel", action="store_true",
                        help="Keep only structural_novelty=novel rows")

    # Dedup / ranking
    parser.add_argument("--dedup", default="none",
                        choices=list(DEDUP_STRATEGIES.keys()),
                        help="Dedup strategy (default: none)")
    parser.add_argument("--top", type=int, default=None,
                        help="Keep only the top N rows after filtering and dedup, "
                             "ranked by apicomplexan_rank (or invasion_probability "
                             "if no panel)")

    # Taxonomy annotation
    parser.add_argument("--taxonomy", default=None,
                        help="User-supplied protein_id -> taxonomy TSV for "
                             "contamination flagging")
    parser.add_argument("--exclude-taxonomy", nargs="+", default=None,
                        help="Drop rows whose lca_taxonomy is in this set "
                             "(requires --taxonomy)")

    # Outputs
    parser.add_argument("--no-html", action="store_true",
                        help="Skip the HTML evidence-card report")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.predictions}...")
    df = pd.read_csv(args.predictions, sep="\t", low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df = apply_filters(df, args)

    if args.dedup != "none":
        before = len(df)
        df = DEDUP_STRATEGIES[args.dedup](df)
        print(f"  Dedup ({args.dedup}): {before} -> {len(df)} rows")

    if args.taxonomy:
        print(f"Loading taxonomy calls from {args.taxonomy}...")
        tax_map = load_taxonomy_calls(args.taxonomy)
        df = annotate_taxonomy(df, tax_map)
        print(f"  lca_taxonomy distribution:")
        for tax, n in df["lca_taxonomy"].value_counts().items():
            print(f"    {tax}: {n}")
        if args.exclude_taxonomy:
            before = len(df)
            df = df[~df["lca_taxonomy"].isin(args.exclude_taxonomy)]
            print(f"  Taxonomy exclusion: {before} -> {len(df)} rows")

    if args.top is not None:
        sort_key = "apicomplexan_rank" if "apicomplexan_rank" in df.columns else "invasion_probability"
        df = df.sort_values(sort_key, ascending=False).head(args.top)
        print(f"  Top {args.top}: {len(df)} rows")

    if len(df) == 0:
        print("  No rows survived filtering; no outputs written.")
        sys.exit(1)

    # Write metadata TSV
    tsv_path = out_dir / "candidates.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Wrote {tsv_path}")

    # Extract FASTA if source provided
    if args.source_fasta:
        print(f"Extracting sequences from {args.source_fasta}...")
        id_to_seq = parse_fasta_ids(args.source_fasta, df["protein_id"].tolist())
        missing = set(df["protein_id"]) - set(id_to_seq.keys())
        if missing:
            print(f"  WARNING: {len(missing)} candidate IDs not found in source FASTA")
        desc_map = dict(zip(df["protein_id"], df.get("description", [""] * len(df))))
        fasta_path = out_dir / "candidates.fasta"
        write_fasta(fasta_path, id_to_seq, desc_map)
        print(f"  Wrote {fasta_path} ({len(id_to_seq)} sequences)")

    # HTML report
    if not args.no_html:
        html_path = out_dir / "candidates.report.html"
        write_html_report(html_path, df, Path(args.predictions).name)
        print(f"  Wrote {html_path}")

    print(f"\nDone. {len(df)} candidates in {out_dir}/")


if __name__ == "__main__":
    main()
