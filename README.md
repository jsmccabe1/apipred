<p align="center">
  <img src="apipred_logo.webp" alt="apipred logo" width="300">
</p>

# apipred

**Predict essential genes and invasion machinery in apicomplexan proteomes from sequence alone.**

apipred uses ESM-2 protein language model embeddings to predict protein essentiality, invasion compartment membership, and structural context for any apicomplexan species, trained on *Toxoplasma gondii* experimental data and validated across *Plasmodium berghei*.

## How it works

```mermaid
graph LR
    A[FASTA proteome] --> B[ESM-2 embedding<br/>1,280 dims per protein]
    B --> C[Essentiality model]
    B --> D[Invasion model]
    B --> E[Reference database]
    C --> F[CRISPR fitness score<br/>Essential / Important / Dispensable]
    D --> G[Invasion probability<br/>Compartment prediction]
    E --> H[Top 3 similar known proteins<br/>Structural novelty flag]
```

## Performance

| Metric | Value |
|--------|-------|
| CRISPR score prediction (Spearman rho) | **0.56** |
| Essential classification (ROC AUC) | **0.77** |
| Invasion classification (ROC AUC) | **0.95** |
| Cross-species transfer to *P. berghei* (Spearman rho) | **0.31** |

## Validation

<p align="center">
  <img src="validation.png" alt="apipred validation" width="700">
</p>

*(A) Cross-species transfer: model trained on T. gondii CRISPR data predicts P. berghei experimental growth rates across 1,136 ortholog pairs (Spearman rho = 0.40). (B) Within-species 5-fold cross-validation on 3,796 T. gondii proteins (rho = 0.56). (C) Per-compartment essentiality: invasion compartments (red) are dispensable in culture while housekeeping machinery (ribosomes, proteasome) is essential, confirming biological coherence.*

## Example output

```bash
python predict.py --input examples/test_proteome.fasta --output predictions.tsv
```

| Protein | Essentiality | CRISPR Score | Invasion | Top Match | Match Compartment |
|---------|-------------|-------------|----------|-----------|-------------------|
| `TGME49_250340` | important | -2.49 | yes | centrin 2 | apical 2 |
| `TGME49_243250` | important | -2.41 | no | myosin H | apical 2 |
| `TGME49_226220` | important | -1.71 | yes | alveolin domain protein | IMC |
| `TGME49_246930` | important | -1.63 | yes | calmodulin CAM1 | apical 2 |
| `TGME49_265790` | important | -1.42 | yes | hypothetical protein | micronemes |
| `TGME49_300100` | dispensable | -0.61 | yes | RON2 | rhoptries 1 |
| `TGME49_262730` | dispensable | 0.46 | yes | ROP16 | rhoptries 1 |

Each protein also gets its top 3 structurally similar characterised proteins (with descriptions, compartments, and CRISPR scores) and a structural novelty flag.

## Installation

```bash
git clone https://github.com/jsmccabe1/apipred.git
cd apipred
pip install -r requirements.txt
```

## Quick start

```bash
# Predict with pre-trained models
python predict.py --input my_proteome.fasta --output predictions.tsv

# Use GPU (recommended for >100 proteins)
python predict.py --input my_proteome.fasta --output predictions.tsv --device cuda
```

## Training from scratch

To retrain models on your own *T. gondii* data (or to regenerate the pre-trained models):

```bash
python train_model.py --data-dir /path/to/Apicomplexa/
```

This generates three files in `models/`:
- `essentiality_model.joblib` - CRISPR score regressor + essential classifier
- `invasion_model.joblib` - invasion compartment classifier
- `reference_db.npz` - 2,634 characterised *T. gondii* proteins for structural context

## Output columns

| Column | Description |
|--------|-------------|
| `protein_id` | FASTA header ID |
| `description` | FASTA header description |
| `length` | Protein length (aa) |
| `predicted_crispr_score` | Predicted CRISPR fitness (more negative = more essential) |
| `essential_probability` | P(essential), where essential = CRISPR score < -3 |
| `essentiality_class` | `essential` / `important` / `dispensable` |
| `invasion_probability` | P(invasion compartment) |
| `predicted_invasion` | `yes` / `no` |
| `similar_1_id` | Most structurally similar characterised protein |
| `similar_1_desc` | Its description |
| `similar_1_compartment` | Its subcellular compartment |
| `similar_1_similarity` | Cosine similarity (0-1) |
| `similar_1_crispr` | Its experimental CRISPR score |
| `similar_2_*`, `similar_3_*` | 2nd and 3rd most similar |
| `max_similarity_to_known` | Highest similarity to any characterised protein |
| `structural_novelty` | `novel` (sim < 0.95) or `known_fold` |

## Training data

- **Essentiality:** Sidik et al. 2016, genome-wide CRISPR screen in *T. gondii* (3,796 proteins)
- **Invasion:** Barylyuk et al. 2020, hyperLOPIT spatial proteomics (425 invasion, 2,209 non-invasion)
- **Embeddings:** ESM-2 (esm2_t33_650M_UR50D, 650M parameters)
- **Models:** Gradient boosting (scikit-learn), 5-fold cross-validated

## Limitations

- Trained on *T. gondii* tachyzoite stage; essentiality may differ in other life stages or species
- Cross-species transfer validated for *P. berghei* blood stages only
- ESM-2 context window is 1,022 tokens; longer proteins use sliding window mean-pooling
- Structural context is relative to characterised *T. gondii* proteins; truly novel folds may be missed
- Invasion predictions trained on hyperLOPIT compartment labels; coarser for non-*Toxoplasma* species

## Citation

If you use apipred, please cite:

> McCabe J et al. (2026) Protein language model embeddings reveal the structural logic of apicomplexan host cell invasion. *In preparation.*

And the underlying methods:
- Lin Z et al. (2023) Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123-1130.
- Sidik SM et al. (2016) A genome-wide CRISPR screen in Toxoplasma identifies essential apicomplexan genes. *Cell* 167:1423-1435.
- Barylyuk K et al. (2020) A comprehensive subcellular atlas of the Toxoplasma proteome via hyperLOPIT. *Cell Host & Microbe* 28:752-766.

## License

MIT
