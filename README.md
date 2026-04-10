# Flow-Faithful Supernode Abstraction for LLM Circuit Tracing

A framework for grouping raw circuit nodes into interpretable **supernodes** while certifying that the coarsened graph faithfully preserves the original attribution flow.

---

## Overview

Circuit tracing on large language models produces attribution graphs with hundreds to thousands of nodes — far too many for direct human inspection. This toolkit abstracts those graphs into small **supernode graphs** (typically 5–10 concept-level nodes) using three principled stages:

1. **Structural similarity clustering** — nodes sharing similar weighted in/out neighbor profiles are grouped via spectral clustering on a cosine-similarity matrix derived from the attribution adjacency.
2. **DAG-constrained layer alignment** — each supernode is constrained to span at most Δ consecutive transformer layers, preserving the layer-wise causal order of the model.
3. **Flow faithfulness certification** — three metrics certify how well the abstraction preserves the original circuit's causal narrative, and their weighted combination guides automatic selection of the number of supernodes *K*.

### Key design invariants

All supernode quantities are **exact partitions** of the original graph — no approximation:

| Quantity | Definition | Conservation |
|---|---|---|
| `sn_inf[S]` | Σ adj[i, logit] for i ∈ S | Σ sn_inf = total logit flow |
| `sn_adj[A,B]` | Σ adj[i,j] for i∈A, j∈B, layer(i)≤layer(j) | Σ sn_adj = total cross-SN forward flow |
| `sn_act[S]` | max activation(i) for i ∈ S | — |

---

## Repository Structure

```
├── structure_grouping.py      # Core: similarity, clustering, DAG enforcement, SN graph
├── auto_grouping.py           # Auto-K selection via eigengap + composite scoring
├── flow_analysis.py           # Flow faithfulness metrics and enhanced auto-K search
├── visualize_circuit_sp_rep.py  # HTML visualization of the supernode circuit
└── subgraph/
    ├── austin_plt.pt          # Example: 65-node factual recall circuit (Gemma-2-2B)
    └── austin_clt.pt          # Example: 22-node pruned circuit variant
```

---

## Setup

### Requirements

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

Python 3.9+ recommended. No GPU required — all operations are on CPU.

### Input format

Each `.pt` file is a PyTorch checkpoint containing:

```python
{
    'kept_ids':    list[str],          # node IDs, last entry is the logit node
    'pruned_adj':  torch.Tensor,       # (N, N) attribution adjacency (receiver-indexed)
    'attr':        dict[str, dict],    # per-node metadata: activation, influence, clerp, ...
    'node_inf':    torch.Tensor,       # (N,) optional normalised influence
}
```

Node ID conventions: `{layer}_{feature}_{ctx}` for middle nodes, `E_{feature}_{ctx}` for embeddings, `27_{feature}_{ctx}` for logit outputs.

---

## Usage

### Quick start — single run with fixed K

```bash
python structure_grouping.py --file subgraph/austin_plt.pt --target-k 7
```

### Auto-K search with flow-enhanced scoring

```bash
python auto_grouping.py --file subgraph/austin_plt.pt --run-best
```

### Full pipeline: flow analysis + visualization

```bash
# Step 1: Run flow-faithful auto-K search, export supernode map + flow JSON
python flow_analysis.py \
    --file subgraph/austin_plt.pt \
    --auto-k --k-min 3 --k-max 5 \
    --alpha 0.7 --beta 0.3 \
    --max-sn 7 \
    --out-json flow_analysis.json

# Step 2: Render the supernode circuit as interactive HTML
python visualize_circuit_sp_rep.py \
    --file subgraph/austin_plt.pt \
    --supernode flow_analysis_supernode_map.json \
    --sn-flow flow_analysis_sn_flow.json \
    --out circuit_sp_rep.html && open circuit_sp_rep.html
```

### Synthetic data (no `.pt` file needed)

```bash
python flow_analysis.py --synthetic --target-k 7
```

---

## Key Parameters

| Flag | Default | Description |
|---|---|---|
| `--target-k` | — | Number of middle supernodes (required for fixed-K mode) |
| `--auto-k` | off | Run sweeping K search |
| `--k-min`, `--k-max` | eigengap±2 | Search range for auto-K |
| `--max-sn` | — | Hard cap on **total** supernodes (fixed EMB/LOGIT nodes counted separately) |
| `--alpha`, `--beta` | 0.5, 0.5 | Weights for activation vs. influence in similarity matrix W |
| `--max-layer-span` | 4 | Maximum transformer layers a single supernode may span |
| `--top-k-paths` | 10 | Number of paths to display in path decomposition report |

---

## Flow Faithfulness Metrics

After clustering, three metrics certify abstraction quality. All are computed from the supernode graph without access to ground-truth labels.

### D(φ) — Flow Distortion

Fraction of total attributed flow **not** captured by the top-K supernode paths from embedding to logit.

```
D(φ) = 1 − (flow in top-k paths) / (total flow)
```

- D(φ) ≈ 0 → a small number of concept-level causal paths explain most of the attribution  
- D(φ) ≈ 1 → flow is diffuse with no dominant narrative

### R(φ) — Local Flow Residual

At each middle supernode S, measures how well in-flow ≈ out-flow + direct logit exit:

```
r_S = |in_flow(S) − out_flow(S)| / (|in_flow(S)| + ε)
R(φ) = mean over all middle supernodes
```

- R(φ) ≈ 0 → the grouping respects flow roles; nodes within each concept have similar flow profiles  
- R(φ) > 1 at a supernode → that supernode merges nodes with fundamentally different roles (e.g., a relay + a suppressive direct writer)

Supernodes with `sn_inf < 0` are flagged as **suppressive concept nodes** — they inhibit the target prediction.

### σ(φ) — Shortcut Fraction

For each direct edge A→C, computes whether an indirect path A→B→C carries comparable flow:

```
ρ(A, C) = w(A→C) / (w(A→C) + max_B min(w(A→B), w(B→C)))
```

- ρ ≈ 1 → direct edge is a genuine concept-to-concept link  
- ρ ≈ 0 → signal flows through intermediaries; the direct edge is a coarse-grouping artifact

σ(φ) = fraction of total edge weight on shortcut edges (ρ < 0.5).

### F(φ) — Combined Score

```
F(φ) = 0.4 × (1 − D(φ)) + 0.3 × (1 − R(φ)) + 0.3 × (1 − σ(φ))
```

| F(φ) | Interpretation |
|---|---|
| > 0.8 | **Excellent** — faithful flow abstraction |
| 0.6–0.8 | **Good** — main flow structure captured with some leakage |
| 0.4–0.6 | **Fair** — consider adjusting K or layer span |
| < 0.4 | **Poor** — grouping too coarse |

---

## Example Results

### `austin_plt.pt` — 65-node circuit, Gemma-2-2B factual recall ("Austin")

```
Best k = 5  →  3 middle supernodes + 4 fixed (EMB × 3, LOGIT × 1) = 7 total
F(φ) = 0.8224  [EXCELLENT]
  path_score     = 0.9949   (top-10 paths capture 99.5% of flow)
  residual_score = 0.5643
  shortcut_score = 0.8506
inf_conservation  = 1.000000
edge_conservation = 1.000000
```

**Top causal paths:**

| Rank | Flow | % | Path |
|---|---|---|---|
| 1 | 18.85 | 30.9% | EMB(" Dallas") → LOGIT(" Austin") |
| 2 | 13.16 | 21.5% | EMB(" Dallas") → SN_01_L19_22 → LOGIT |
| 3 | 8.60 | 14.1% | EMB(" Dallas") → SN_00_L16_19 → SN_01_L19_22 → LOGIT |
| 4 | 6.22 | 10.2% | EMB(" capital") → SN_01_L19_22 → LOGIT |

The dominant narrative: the " Dallas" embedding activates a chain of late-layer city/state concept nodes (L16–22) which write directly to the " Austin" logit. The " capital" embedding provides supporting signal through the same pathway.

**Suppressive node detected:** `SN_02_L22_25` has `inf_exit = −2.25` (inhibits competing predictions) with residual `r = 1.006` — it receives large in-flow but routes almost none to the logit, acting as a flow sink for suppressed alternatives.

---

### `austin_clt.pt` — 22-node pruned circuit

```
Best k = 3  →  3 middle supernodes + 4 fixed = 7 total
F(φ) = 0.7811  [GOOD]
  path_score     = 1.0000   (all 7 paths captured in top-10)
  residual_score = 0.3603
  shortcut_score = 0.9100
```

**Top causal paths:**

| Rank | Flow | % | Path |
|---|---|---|---|
| 1 | 35.56 | 45.6% | EMB(" Dallas") → SN_02_L16_22 → LOGIT |
| 2 | 18.63 | 23.9% | EMB(" Dallas") → LOGIT (direct) |
| 3 | 11.78 | 15.1% | EMB(" capital") → SN_02_L16_22 → LOGIT |

The pruned circuit concentrates flow into fewer paths with the direct EMB→LOGIT edge carrying nearly a quarter of all attribution, consistent with the embedding layer already encoding strong prior probability for " Austin".

---

## Outputs

After a full pipeline run, the following files are produced:

| File | Contents |
|---|---|
| `flow_analysis.json` | Full flow faithfulness report (paths, residuals, shortcuts, scores) |
| `flow_analysis_sn_flow.json` | Supernode graph quantities for the visualizer |
| `flow_analysis_supernode_map.json` | Mapping from supernode name → list of member node IDs |
| `circuit_sp_rep.html` | Interactive HTML circuit visualization |
| `auto_k_results.json` | K-sweep scores (from `auto_grouping.py`) |
| `auto_k_plot.png` | Eigengap + composite score plot |

---

## Theoretical Background

The method is grounded in the **Layered Degree-Corrected Block Model (LDCBM)**: under this generative model for attribution graphs, nodes in the same concept block have identical structural equivalence class. Optimizing for structural similarity therefore simultaneously optimizes for flow preservation.

**Formal guarantees** (see `method.tex` for full proofs):

- **Concept recovery** (Theorem 1): spectral clustering recovers the true concept partition with high probability when SNR ≥ C√(log N / d̄).
- **Flow distortion bound** (Theorem 2): D(φ) ≤ W_shortcut / W_total — flow distortion is bounded by the fraction of edge weight on shortcut edges.
- **Local conservation bound** (Theorem 3): R(φ) ≤ max δ_k / min d̄_k^inter — residuals are bounded by within-block flow asymmetry, which the structural similarity objective directly minimizes.
- **Path concentration** (Proposition 4): with supernode out-degree d and DAG depth L, the number of significant paths is O(d^L), consistent with empirically observed near-zero D(φ).

---

## Citation

If you use this code, please cite the accompanying paper (`method.tex`):

```
Flow-Faithful Supernode Abstraction for Mechanistic Interpretation of LLM Circuits
```