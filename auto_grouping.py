"""
auto_grouping.py
─────────────────────────────────────────────────────────────────────────────
Automatic selection of the optimal target-k for structure_grouping.py.

Combines two strategies:
  1. Eigengap heuristic    → narrows the search range
  2. Flow-aware objective  → picks the best k within that range

Usage (standalone):
  python auto_k.py --file subgraph/austin_plt.pt --max-layer-span 10
  python auto_k.py --file subgraph/austin_clt.pt --max-layer-span 10
(eeg) anhnd@Ducs-MacBook-Pro-2 XCIRCUIT % >....


python visualize_circuit_sp_rep.py \
    --file subgraph/austin_clt.pt \
    --supernode supernode_map.json \
    --sn-flow supernode_map_sn_flow.json \
    --out circuit_sp_rep.html

open circuit_sp_rep.html

Loading subgraph/austin_clt.pt...
  Nodes: 22
Computing similarity matrix...
  S shape: torch.Size([22, 22])

── Step 1: Eigengap analysis ──
  Eigengap suggests k = 2
  Eigenvalue gaps (first 10): 0.1056, 0.5465, 0.2428, 0.0271, 0.0112, 0.0111, 0.0142, 0.0129, 0.0061, 0.0119
  Initial search range: [2, 4]
  Final search range: [3, 5]

── Step 2: Scoring k = 3..5 ──
    k  n_sn   intra    dag  flow/attr   size   TOTAL  warns
  ──────────────────────────────────────────────────
  Similarity stats (middle nodes, upper triangle):
    mean=0.3728  median=0.3783  p75=0.6607  max=1.0000
  Spectral clusters (before DAG enforcement): 3
    3     9  1.0000  1.0000  0.6643  0.7222  0.8605      0
  Similarity stats (middle nodes, upper triangle):
    mean=0.3728  median=0.3783  p75=0.6607  max=1.0000
  Spectral clusters (before DAG enforcement): 4
    4    11  1.0000  1.0000  0.6641  0.6111  0.8383      0
  Similarity stats (middle nodes, upper triangle):
    mean=0.3728  median=0.3783  p75=0.6607  max=1.0000
  Spectral clusters (before DAG enforcement): 5
    5    11  1.0000  1.0000  0.6631  0.6111  0.8380      0

── Result ──
  Best k = 3  (total score = 0.8605)
    intra_sim    = 1.0000  (within-cluster coherence)
    dag_safety   = 1.0000  (cycle-free fraction)
    flow_balance = 0.6643  (entropy of flow distribution)
    size_score   = 0.7222  (proximity to sqrt(N)=4)
    n_middle supernodes after DAG enforcement = 9
    DAG cycle warnings = 0

Sweep results saved → auto_k_results.json

Auto-k plot saved → auto_k_plot.png

── Running structure_grouping with k=3 ──
Supernode map saved → supernode_map.json

════════════════════════════════════════════════════════════════════════
  STRUCTURE GROUPING REPORT
════════════════════════════════════════════════════════════════════════

  Total supernodes  : 11
  Fixed (kept)      : SN_EMB, SN_LOGIT
  Clustered         : 9

────────────────────────────────────────────────────────────────────────
  SUPERNODE SUMMARY
────────────────────────────────────────────────────────────────────────
  Name                     n      layers   act_sum   inf_sum   intra_sim
  --------------------------------------------------------------------
  SN_00_L1                 1          L1      9.19    0.4841      1.0000
  SN_01_L7                 1          L7      3.52    0.4605      1.0000
  SN_02_L16                1         L16      2.41    0.4011      1.0000
  SN_03_L17                2         L17      4.56    0.9913      0.7236
  SN_04_L18                1         L18      2.27    0.5065      1.0000
  SN_05_L18_19             3      L18–19      5.96    1.4743      0.7738
  SN_06_L20_22             3      L20–22     11.64    1.2133      0.8225
  SN_07_L20_21             3      L20–21     11.06    1.3164      0.7173
  SN_08_L22                3         L22      6.69    1.4559      0.9062
  SN_EMB                   3          L0     64.85    0.6485      0.1118
  SN_LOGIT                 1         L27     43.95    0.0000      1.0000

────────────────────────────────────────────────────────────────────────
  MEMBER DETAILS
────────────────────────────────────────────────────────────────────────

  ┌── SN_00_L1  (n=1, layers L1–1, intra_sim_min=1.000)
  │   1_89326_9             L1    Dallas
  └────────────────────────────────────────────────────────────

  ┌── SN_01_L7  (n=1, layers L7–7, intra_sim_min=1.000)
  │   7_24743_9             L7    Texas legal matters
  └────────────────────────────────────────────────────────────

  ┌── SN_02_L16  (n=1, layers L16–16, intra_sim_min=1.000)
  │   16_89970_9            L16   Texas
  └────────────────────────────────────────────────────────────

  ┌── SN_03_L17  (n=2, layers L17–17, intra_sim_min=0.724)
  │   17_1822_10            L17   place names
  │   17_98126_10           L17   Government buildings/institutions
  └────────────────────────────────────────────────────────────

  ┌── SN_04_L18  (n=1, layers L18–18, intra_sim_min=1.000)
  │   18_61980_10           L18   Texas related
  └────────────────────────────────────────────────────────────

  ┌── SN_05_L18_19  (n=3, layers L18–19, intra_sim_min=0.661)
  │   18_3623_10            L18   capital cities
  │   18_45367_10           L18   US cities
  │   19_54790_10           L19   Places
  └────────────────────────────────────────────────────────────

  ┌── SN_06_L20_22  (n=3, layers L20–22, intra_sim_min=0.740)
  │   20_44686_9            L20   Texas locations/legal contexts
  │   20_44686_10           L20   Texas locations/legal contexts
  │   22_32893_10           L22   Texas legal documents
  └────────────────────────────────────────────────────────────

  ┌── SN_07_L20_21  (n=3, layers L20–21, intra_sim_min=0.690)
  │   20_74108_10           L20   capital
  │   21_61721_10           L21   code/documentation
  │   21_84338_10           L21   Cities and locations
  └────────────────────────────────────────────────────────────

  ┌── SN_08_L22  (n=3, layers L22–22, intra_sim_min=0.827)
  │   22_11998_10           L22   Texas and Dallas
  │   22_74186_10           L22   general news snippets
  │   22_81571_10           L22   Texas locations and organizations
  └────────────────────────────────────────────────────────────

  ┌── SN_EMB  (n=3, layers L0–0, intra_sim_min=0.000)
  │   E_6037_4              L0    Emb: "  capital"
  │   E_2329_7              L0    Emb: "  state"
  │   E_26865_9             L0    Emb: "  Dallas"
  └────────────────────────────────────────────────────────────

  ┌── SN_LOGIT  (n=1, layers L27–27, intra_sim_min=1.000)
  │   27_22605_10           L27   Output " Austin" (p=0.439)
  └────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────
  FLOW PRESERVATION  (legacy: node-level reach partitioned per SN)
────────────────────────────────────────────────────────────────────────
  NOTE: ratio is trivially 1.0 — see SUPERNODE FLOW REPORT for
        a meaningful surrogate comparison.
  Original  reach → logit : 2.5234  (normalised)
  Surrogate reach → logit : 2.5234
  Preservation ratio      : 1.0000
  [PASS] Within 20% tolerance.

  Structural reach → logit (by supernode):
    SN_00_L1                -0.0242  ( -1.0%)
    SN_01_L7                 0.0708  (  2.8%)  █
    SN_02_L16                0.0681  (  2.7%)  █
    SN_03_L17                0.0674  (  2.7%)  █
    SN_04_L18               -0.0270  ( -1.1%)
    SN_05_L18_19             0.0958  (  3.8%)  █
    SN_06_L20_22             0.3720  ( 14.7%)  ███████
    SN_07_L20_21             0.2424  (  9.6%)  ████
    SN_08_L22               -0.1690  ( -6.7%)
    SN_EMB                   1.8271  ( 72.4%)  ████████████████████████████████████

────────────────────────────────────────────────────────────────────────
  ATTRIBUTION SUMMARY  (causal influence scores, precomputed)
────────────────────────────────────────────────────────────────────────
  (Captures total causal effect per supernode on logit,
   independent of path — NOT an edge, a node property.)

    SN_00_L1                 0.4841  (  5.4%)  ██
    SN_01_L7                 0.4605  (  5.1%)  ██
    SN_02_L16                0.4011  (  4.5%)  ██
    SN_03_L17                0.9913  ( 11.1%)  █████
    SN_04_L18                0.5065  (  5.7%)  ██
    SN_05_L18_19             1.4743  ( 16.5%)  ████████
    SN_06_L20_22             1.2133  ( 13.6%)  ██████
    SN_07_L20_21             1.3164  ( 14.7%)  ███████
    SN_08_L22                1.4559  ( 16.3%)  ████████
    SN_EMB                   0.6485  (  7.2%)  ███

────────────────────────────────────────────────────────────────────────
  DAG SAFETY CHECK
────────────────────────────────────────────────────────────────────────
  [PASS] No interleaving layer ranges detected.

════════════════════════════════════════════════════════════════════════

DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 0.0000
DEBUG SN_EMB incoming (col sum): 0.0000
DEBUG SN_EMB outgoing (row sum): 52.7041
DEBUG SN_EMB incoming (col sum): 0.0000
  EMB → SN_00_L1: 9.6875
  EMB → SN_01_L7: 5.1250
  EMB → SN_02_L16: 3.6396
  EMB → SN_03_L17: 2.8125
  EMB → SN_04_L18: 2.2344
  EMB → SN_05_L18_19: 2.3809
  EMB → SN_06_L20_22: 11.3047
  EMB → SN_07_L20_21: 3.5703
  EMB → SN_08_L22: 7.3320
  EMB → SN_LOGIT: 4.6172
DEBUG SN_EMB outgoing (row sum): 52.7041
DEBUG SN_EMB incoming (col sum): 0.0000
  EMB → SN_00_L1: 9.6875
  EMB → SN_01_L7: 5.1250
  EMB → SN_02_L16: 3.6396
  EMB → SN_03_L17: 2.8125
  EMB → SN_04_L18: 2.2344
  EMB → SN_05_L18_19: 2.3809
  EMB → SN_06_L20_22: 11.3047
  EMB → SN_07_L20_21: 3.5703
  EMB → SN_08_L22: 7.3320
  EMB → SN_LOGIT: 4.6172

════════════════════════════════════════════════════════════════════════
  SUPERNODE FLOW REPORT  (surrogate graph, SN-level propagation)
════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────
  FLOW PRESERVATION  (SN surrogate vs. original node graph)
────────────────────────────────────────────────────────────────────────
  Original node-level reach → logit : 9.2636
  Surrogate SN-level reach  → logit : 2.9060
  Preservation ratio                : 0.3137
  [FAIL] Ratio 0.314 — SN graph is a poor surrogate.

────────────────────────────────────────────────────────────────────────
  PER-SUPERNODE REACH → SN_LOGIT  (multi-hop, SN graph)
────────────────────────────────────────────────────────────────────────
  Supernode               act_norm       reach   share%  flow
  SN_00_L1                  0.2091     -1.6918   -58.2%
  SN_01_L7                  0.0800      0.5295    18.2%  █████████
  SN_02_L16                 0.0548      0.3738    12.9%  ██████
  SN_03_L17                 0.0519      0.6698    23.0%  ███████████
  SN_04_L18                 0.0516     -0.0270    -0.9%
  SN_05_L18_19              0.0452      0.2658     9.1%  ████
  SN_06_L20_22              0.0883      0.1944     6.7%  ███
  SN_07_L20_21              0.0839      0.4308    14.8%  ███████
  SN_08_L22                 0.0507     -0.1107    -3.8%
  SN_EMB                    0.4919      2.2712    78.2%  ███████████████████████████████████████

────────────────────────────────────────────────────────────────────────
  DOMINANT SN→SN FLOW EDGES  (top 5 by F_sn weight)
────────────────────────────────────────────────────────────────────────
  Source                  Target                   F_sn weight
  SN_EMB                  SN_06_L20_22                5.560860
  SN_EMB                  SN_00_L1                    4.765353
  SN_EMB                  SN_08_L22                   3.606681
  SN_EMB                  SN_01_L7                    2.521026
  SN_EMB                  SN_LOGIT                    2.271229

────────────────────────────────────────────────────────────────────────
  HOP-BY-HOP FLOW TO SN_LOGIT
────────────────────────────────────────────────────────────────────────
  Hop 1: total=3.0078  (top contributors: SN_EMB=2.2712, SN_06_L20_22=0.3739, SN_07_L20_21=0.2340)
  Hop 2: total=0.5370  (top contributors: SN_01_L7=0.4506, SN_02_L16=0.3849, SN_03_L17=0.2417)

────────────────────────────────────────────────────────────────────────
  BOTTLENECK SUPERNODES  (high in-flow, low out-flow)
────────────────────────────────────────────────────────────────────────
  SN_00_L1                in=4.7654  out=-0.3169  ratio=-0.066
  SN_08_L22               in=3.8712  out=-0.1107  ratio=-0.029

────────────────────────────────────────────────────────────────────────
  SN→SN ADJACENCY MATRIX  (raw edge weights, non-zero only)
────────────────────────────────────────────────────────────────────────
  SN_01_L7               → SN_02_L16:0.480, SN_06_L20_22:1.207, SN_LOGIT:0.688
  SN_02_L16              → SN_06_L20_22:1.891, SN_08_L22:0.447, SN_LOGIT:1.086
  SN_03_L17              → SN_05_L18_19:1.188, SN_07_L20_21:0.805, SN_LOGIT:1.156
  SN_05_L18_19           → SN_07_L20_21:0.977, SN_LOGIT:2.031
  SN_06_L20_22           → SN_07_L20_21:0.988, SN_08_L22:2.719, SN_LOGIT:4.234
  SN_07_L20_21           → SN_06_L20_22:0.660, SN_LOGIT:2.789
  SN_EMB                 → SN_00_L1:9.688, SN_01_L7:5.125, SN_02_L16:3.640, SN_03_L17:2.812, SN_04_L18:2.234, SN_05_L18_19:2.381, SN_06_L20_22:11.305, SN_07_L20_21:3.570, SN_08_L22:7.332, SN_LOGIT:4.617

════════════════════════════════════════════════════════════════════════

Supernode flow saved → supernode_map_sn_flow.json

  ➜  Use:  python auto_grouping.py --file subgraph/austin_clt.pt --target-k 3 --max-layer-span 4
Loading subgraph/austin_clt.pt...
  Total edges:    61
  Edges to logit (real circuit only): 20
  Nodes: 22
  Edges: 61
Loading supernode map: supernode_map.json...
  Supernodes: 11
Loading SN flow data: supernode_map_sn_flow.json...
  SN flow loaded: preservation=0.3137
Saved → circuit_sp_rep.html
Open:  open circuit_sp_rep.html
(eeg) anhnd@Ducs-MacBook-Pro-2 XCIRCUIT %
Usage (as library):
  from auto_k import find_best_k
  best_k, results = find_best_k(data, S, max_layer_span=10)

Then pass best_k to:
  python structure_grouping.py --file ... --target-k <best_k>
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.linalg import eigvalsh

# ── Import from structure_grouping ────────────────────────────────────────────
from structure_grouping import (
    load_snapshot,
    prepare_graph_data,
    compute_similarity,
    cluster_with_target_k,
    check_dag_safety,
    evaluate_grouping,
    compute_surrogate_flow,
    parse_layer,
    _is_fixed,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EIGENGAP HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

def eigengap_analysis(S: torch.Tensor,
                      kept_ids: list,
                      max_k: int = 20) -> dict:
    """
    Compute eigenvalues of the normalized graph Laplacian of the
    middle-node similarity submatrix. The largest gap between consecutive
    eigenvalues suggests the natural number of clusters.

    Returns:
        eigengap_k    : int         suggested k from largest gap
        eigenvalues   : np.ndarray  first max_k eigenvalues (sorted ascending)
        gaps          : np.ndarray  gaps[i] = eigenvalues[i+1] - eigenvalues[i]
        search_range  : (int, int)  recommended k_min, k_max for sweep
    """
    # Extract middle-node submatrix
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    M = len(middle_idx)

    if M < 3:
        return dict(eigengap_k=2, eigenvalues=np.array([0, 1]),
                    gaps=np.array([1]), search_range=(2, 2))

    S_mid = S[middle_idx][:, middle_idx].numpy()
    S_mid = ((S_mid + S_mid.T) / 2).clip(0.0, 1.0)

    # Degree matrix and normalized Laplacian
    deg = S_mid.sum(axis=1)
    deg_safe = np.where(deg > 1e-8, deg, 1e-8)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))

    L = np.diag(deg) - S_mid
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute eigenvalues (symmetric → eigvalsh is stable)
    n_eig = min(max_k + 1, M)
    eigenvalues = np.sort(eigvalsh(L_norm))[:n_eig]

    # Find gaps (skip the trivial first eigenvalue ≈ 0)
    # We look at gaps starting from index 1
    # gap[i] = λ_{i+1} - λ_i, suggesting i+1 clusters
    gaps = np.diff(eigenvalues)

    # The k suggested by eigengap: largest gap in range [1, max_k-1]
    # gap at index i suggests k = i + 1 clusters
    search_end = min(len(gaps), max_k)
    if search_end < 2:
        eigengap_k = 2
    else:
        # Skip gap[0] (between λ_0 and λ_1) — it's always large
        gap_slice = gaps[1:search_end]
        eigengap_k = int(np.argmax(gap_slice)) + 2  # +2 because we skipped index 0

    # Search range: ±2 around eigengap suggestion, clamped
    k_min = max(2, eigengap_k - 2)
    k_max = min(M - 1, eigengap_k + 2)

    # Ensure range is at least 3 wide for meaningful comparison
    if k_max - k_min < 2:
        k_max = min(M - 1, k_min + 4)

    return dict(
        eigengap_k   = eigengap_k,
        eigenvalues  = eigenvalues,
        gaps         = gaps,
        search_range = (k_min, k_max),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FLOW-AWARE COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_k(final_supernodes: dict,
            data: dict,
            S: torch.Tensor,
            target_n_middle: int,
            w_intra: float = 0.30,
            w_dag:   float = 0.25,
            w_flow:  float = 0.25,
            w_size:  float = 0.20) -> dict:
    """
    Evaluate a supernode grouping with a composite score.

    Components (all normalised to [0, 1]):
      1. intra_sim   : mean intra-cluster similarity (higher = better)
      2. dag_safety  : 1 - (fraction of SN pairs with cycle warnings)
      3. flow_balance: entropy of flow distribution / log(n_sn)
                       (uniform → 1.0, single-path → 0.0)
      4. size_penalty: 1 - |n_middle - sqrt(N_middle_nodes)| / N_middle_nodes
                       penalises both too many and too few supernodes

    Returns:
        dict with component scores and total
    """
    stats       = evaluate_grouping(final_supernodes, data, S)
    flow_result = compute_surrogate_flow(final_supernodes, data)
    dag_warnings = check_dag_safety(final_supernodes)

    # ── 1. Intra-cluster similarity (exclude EMB/LOGIT — they're fixed)
    middle_sn = {sn: st for sn, st in stats.items()
                 if sn not in ('SN_EMB', 'SN_LOGIT')}
    n_middle = len(middle_sn)

    if n_middle == 0:
        return dict(total=0, intra_sim=0, dag_safety=0, flow_balance=0,
                    size_score=0, n_middle=0, details={})

    intra_sims = [st['intra_sim_mean'] for st in middle_sn.values()]
    sizes = [st['n'] for st in middle_sn.values()]
    total_n = sum(sizes)
    intra_sim_raw = sum(s * n / total_n for s, n in zip(intra_sims, sizes))

    # Normalise against the global mean similarity so scores are comparable
    # across graphs with very different similarity distributions
    middle_idx = [i for i, nid in enumerate(data['kept_ids']) if not _is_fixed(nid)]
    S_mid = S[middle_idx][:, middle_idx].numpy()
    upper = S_mid[np.triu_indices(len(middle_idx), k=1)]
    global_mean = float(upper.mean()) + 1e-8
    # Ratio > 1 means clusters are more similar than random pairs
    intra_sim = min(1.0, intra_sim_raw / (global_mean * 2))

    # ── 2. DAG safety
    n_pairs = n_middle * (n_middle - 1) / 2
    if n_pairs > 0:
        dag_safety = 1.0 - len(dag_warnings) / n_pairs
    else:
        dag_safety = 1.0

    # ── 3. Flow balance
    sn_flow = flow_result['sn_flow']
    flow_vals = []
    for src in final_supernodes:
        if src == 'SN_LOGIT':
            continue
        val = sn_flow.get(src, {}).get('SN_LOGIT', 0.0)
        flow_vals.append(max(val, 0.0))

    total_flow = sum(flow_vals) + 1e-8

    # ── NEW: fall back to attribution distribution if no structural flow to logit
    if total_flow < 1e-6:
        attr_vals = [
            max(flow_result['sn_attribution'].get(src, 0.0), 0.0)
            for src in final_supernodes
            if src != 'SN_LOGIT'
        ]
        total_attr = sum(attr_vals) + 1e-8
        probs = np.array([v / total_attr for v in attr_vals])
    else:
        probs = np.array([v / total_flow for v in flow_vals])

    probs = probs[probs > 1e-10]

    if len(probs) > 1:
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        flow_balance = entropy / (max_entropy + 1e-8)
    else:
        flow_balance = 0.0
    # ── 4. Size score: prefer k near sqrt(N_middle_nodes)

    # Better — penalise DAG over-splitting too
        # ── 4. Size score: prefer k near sqrt(N_middle_nodes)
    ideal_k = max(2, int(np.sqrt(target_n_middle)))
    deviation = abs(n_middle - ideal_k) / max(target_n_middle, 1)
    size_score = max(0.0, 1.0 - deviation)

    # ── 4b. Cohesion penalty: punish clusters with very low intra_sim_min
    weak_clusters = sum(1 for st in middle_sn.values()
                        if st['intra_sim_min'] < 0.3)
    cohesion_penalty = weak_clusters / max(n_middle, 1)
    intra_sim = intra_sim * (1.0 - 0.5 * cohesion_penalty)
    size_score = max(0.0, 1.0 - deviation)

    # ── Composite
    total = (w_intra * intra_sim
             + w_dag * dag_safety
             + w_flow * flow_balance
             + w_size * size_score)

    return dict(
        total        = total,
        intra_sim    = intra_sim,
        dag_safety   = dag_safety,
        flow_balance = flow_balance,
        size_score   = size_score,
        n_middle     = n_middle,
        n_warnings   = len(dag_warnings),
        flow_ratio   = flow_result['ratio'],
        details      = {sn: {'intra_sim': st['intra_sim_mean'],
                              'n': st['n'],
                              'layers': f"L{st['layer_lo']}-{st['layer_hi']}"}
                        for sn, st in middle_sn.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN SEARCH: eigengap → sweep → score
# ─────────────────────────────────────────────────────────────────────────────

def find_best_k(data: dict,
                S: torch.Tensor,
                max_layer_span: int = 4,
                k_min_override: int = None,
                k_max_override: int = None,
                weights: dict = None) -> tuple:
    """
    Full auto-k pipeline:
      1. Eigengap heuristic to set search range
      2. For each k in range, run spectral clustering + DAG enforcement
      3. Score each result with composite objective
      4. Return best k and all results

    Args:
        data            : from prepare_graph_data()
        S               : from compute_similarity()
        max_layer_span  : passed to cluster_with_target_k()
        k_min_override  : force lower bound of search (overrides eigengap)
        k_max_override  : force upper bound of search (overrides eigengap)
        weights         : dict with keys w_intra, w_dag, w_flow, w_size

    Returns:
        best_k  : int
        results : dict  k → score_dict
    """
    kept_ids = data['kept_ids']
    middle_ids = [nid for nid in kept_ids if not _is_fixed(nid)]
    N_middle = len(middle_ids)

    if N_middle < 3:
        print('  Too few middle nodes for auto-k. Defaulting to k=2.')
        return 2, {}

    # ── Step 1: Eigengap
    print('\n── Step 1: Eigengap analysis ──')
    eg = eigengap_analysis(S, kept_ids, max_k=min(20, N_middle - 1))
    print(f'  Eigengap suggests k = {eg["eigengap_k"]}')
    print(f'  Eigenvalue gaps (first 10): '
          f'{", ".join(f"{g:.4f}" for g in eg["gaps"][:10])}')
    print(f'  Initial search range: [{eg["search_range"][0]}, {eg["search_range"][1]}]')

    k_min = k_min_override if k_min_override is not None else eg['search_range'][0]
    k_max = k_max_override if k_max_override is not None else eg['search_range'][1]
    k_min = max(2, k_min)
    k_max = min(N_middle - 1, k_max)

    if k_min > k_max:
        k_min = k_max

    print(f'  Final search range: [{k_min}, {k_max}]')

    w = weights or {}

    # ── Step 2: Sweep
    print(f'\n── Step 2: Scoring k = {k_min}..{k_max} ──')
    # In find_best_k, replace the header line:
    print(f'  {"k":>3}  {"n_sn":>4}  {"intra":>6}  {"dag":>5}  '
          f'{"flow/attr":>9}  {"size":>5}  {"TOTAL":>6}  {"warns":>5}')
    print(f'  {"─"*50}')

    results = {}
    for k in range(k_min, k_max + 1):
        try:
            final_sn = cluster_with_target_k(
                data, S, target_k=k, max_layer_span=max_layer_span
            )
        except Exception as e:
            print(f'  k={k} failed: {e}')
            continue

        w = weights or {}
        sc = score_k(
            final_sn, data, S,
            target_n_middle=N_middle,
            w_intra=w.get('w_intra', 0.30),
            w_dag=w.get('w_dag', 0.25),
            w_flow=w.get('w_flow', 0.25),
            w_size=w.get('w_size', 0.20),
        )

        sc['final_supernodes'] = final_sn  # store for --run-best
        results[k] = sc  # ← THIS is what was missing

        print(f'  {k:>3}  {sc["n_middle"]:>4}  {sc["intra_sim"]:>6.4f}  '
              f'{sc["dag_safety"]:>5.4f}  {sc["flow_balance"]:>5.4f}  '
              f'{sc["size_score"]:>5.4f}  {sc["total"]:>6.4f}  '
              f'{sc.get("n_warnings", 0):>5}  ')

    if not results:
        print('  All k values failed. Falling back to eigengap suggestion.')
        return eg['eigengap_k'], {}

    # ── Step 3: Pick best
    best_k = max(results, key=lambda k: results[k]['total'])
    best_score = results[best_k]

    print(f'\n── Result ──')
    print(f'  Best k = {best_k}  (total score = {best_score["total"]:.4f})')
    print(f'    intra_sim    = {best_score["intra_sim"]:.4f}  '
          f'(within-cluster coherence)')
    print(f'    dag_safety   = {best_score["dag_safety"]:.4f}  '
          f'(cycle-free fraction)')
    print(f'    flow_balance = {best_score["flow_balance"]:.4f}  '
          f'(entropy of flow distribution)')
    print(f'    size_score   = {best_score["size_score"]:.4f}  '
          f'(proximity to sqrt(N)={int(np.sqrt(N_middle))})')
    print(f'    n_middle supernodes after DAG enforcement = {best_score["n_middle"]}')
    print(f'    DAG cycle warnings = {best_score["n_warnings"]}')

    return best_k, results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTIONAL: save detailed results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, best_k: int, out_path: str = 'auto_k_results.json'):
    """Save the sweep results for inspection / plotting."""
    out = {
        'best_k': best_k,
        'sweep': {}
    }
    for k, sc in results.items():
        entry = {key: val for key, val in sc.items()
                 if key != 'final_supernodes'}
        out['sweep'][str(k)] = entry

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSweep results saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTIONAL: plot eigengap + score comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_auto_k(eigengap_result: dict, results: dict, best_k: int,
                out_path: str = 'auto_k_plot.png'):
    """Plot eigengap and composite scores side by side."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [INFO] matplotlib not available — skipping plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: eigenvalue gaps
    ax = axes[0]
    gaps = eigengap_result['gaps']
    x_gaps = list(range(2, 2 + len(gaps)))
    ax.bar(x_gaps, gaps, color='#3d8eff', alpha=0.7, edgecolor='#1a2540')
    eg_k = eigengap_result['eigengap_k']
    if eg_k in x_gaps:
        idx = x_gaps.index(eg_k)
        ax.bar(eg_k, gaps[idx], color='#f5a623', alpha=0.9, edgecolor='#1a2540')
    ax.set_xlabel('k (number of clusters)', fontsize=9)
    ax.set_ylabel('Eigenvalue gap (λ_{k+1} - λ_k)', fontsize=9)
    ax.set_title('Eigengap Heuristic', fontsize=11, fontweight='bold')
    ax.axvline(x=eg_k, color='#f5a623', linestyle='--', alpha=0.6,
               label=f'eigengap k={eg_k}')
    ax.legend(fontsize=8)

    # ── Right: composite scores
    ax = axes[1]
    ks = sorted(results.keys())
    totals = [results[k]['total'] for k in ks]
    colors = ['#f5a623' if k == best_k else '#3d8eff' for k in ks]
    ax.bar(ks, totals, color=colors, alpha=0.8, edgecolor='#1a2540')

    # Stacked components
    components = ['intra_sim', 'dag_safety', 'flow_balance', 'size_score']
    comp_colors = ['#10b981', '#8b5cf6', '#f43f5e', '#06b6d4']
    comp_weights = [0.30, 0.25, 0.25, 0.20]
    bottom = np.zeros(len(ks))
    for comp, cc, w in zip(components, comp_colors, comp_weights):
        vals = [results[k][comp] * w for k in ks]
        ax.bar(ks, vals, bottom=bottom, color=cc, alpha=0.3, width=0.6)
        bottom += np.array(vals)

    ax.set_xlabel('k (target clusters)', fontsize=9)
    ax.set_ylabel('Composite score', fontsize=9)
    ax.set_title('Flow-Aware Objective', fontsize=11, fontweight='bold')
    ax.axvline(x=best_k, color='#f5a623', linestyle='--', alpha=0.6,
               label=f'best k={best_k}')

    # Mini legend for components
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10b981', alpha=0.4, label='intra_sim (0.30)'),
        Patch(facecolor='#8b5cf6', alpha=0.4, label='dag_safety (0.25)'),
        Patch(facecolor='#f43f5e', alpha=0.4, label='flow_balance (0.25)'),
        Patch(facecolor='#06b6d4', alpha=0.4, label='size_score (0.20)'),
        Patch(facecolor='#f5a623', alpha=0.9, label=f'best k={best_k}'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'\nAuto-k plot saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automatically determine optimal target-k for structure_grouping.py')
    parser.add_argument('--file',           type=str, required=True,
                        help='Path to circuit subgraph .pt file')
    parser.add_argument('--max-layer-span', type=int, default=10,
                        help='Max layer span per supernode (default 10)')
    parser.add_argument('--alpha',          type=float, default=0.5,
                        help='Activation weight in similarity (default 0.5)')
    parser.add_argument('--beta',           type=float, default=0.5,
                        help='Influence weight in similarity (default 0.5)')
    parser.add_argument('--k-min',          type=int, default=None,
                        help='Override minimum k in search range')
    parser.add_argument('--k-max',          type=int, default=None,
                        help='Override maximum k in search range')
    parser.add_argument('--w-intra',        type=float, default=0.30,
                        help='Weight for intra-cluster similarity (default 0.30)')
    parser.add_argument('--w-dag',          type=float, default=0.25,
                        help='Weight for DAG safety (default 0.25)')
    parser.add_argument('--w-flow',         type=float, default=0.25,
                        help='Weight for flow balance (default 0.25)')
    parser.add_argument('--w-size',         type=float, default=0.20,
                        help='Weight for size penalty (default 0.20)')
    parser.add_argument('--out-json',       type=str, default='auto_k_results.json',
                        help='Output path for sweep results JSON')
    parser.add_argument('--out-plot',       type=str, default='auto_k_plot.png',
                        help='Output path for comparison plot')
    parser.add_argument('--run-best',       action='store_true',
                        help='After finding best k, run structure_grouping and '
                             'save the supernode_map.json')
    args = parser.parse_args()

    # ── Load
    print(f'Loading {args.file}...')
    raw  = load_snapshot(args.file)
    data = prepare_graph_data(raw)
    N    = len(data['kept_ids'])
    print(f'  Nodes: {N}')

    # ── Similarity
    print('Computing similarity matrix...')
    S = compute_similarity(data, alpha=args.alpha, beta=args.beta)
    print(f'  S shape: {S.shape}')

    # ── Eigengap
    eg = eigengap_analysis(S, data['kept_ids'])

    # ── Search
    weights = dict(w_intra=args.w_intra, w_dag=args.w_dag,
                   w_flow=args.w_flow, w_size=args.w_size)

    best_k, results = find_best_k(
        data, S,
        max_layer_span  = args.max_layer_span,
        k_min_override  = args.k_min,
        k_max_override  = args.k_max,
        weights         = weights,
    )

    # ── Save
    if results:
        save_results(results, best_k, args.out_json)
        plot_auto_k(eg, results, best_k, args.out_plot)

    # ── Optionally run the full pipeline with best k
    if args.run_best and best_k in results:
        print(f'\n── Running structure_grouping with k={best_k} ──')
        final_sn = results[best_k]['final_supernodes']

        out_map = {sn: members for sn, members in final_sn.items()}
        with open('supernode_map.json', 'w') as f:
            json.dump(out_map, f, indent=2)
        print(f'Supernode map saved → supernode_map.json')

        from structure_grouping import (
            print_report,
            compute_supernode_flow,
            print_supernode_flow_report,
        )
        stats = evaluate_grouping(final_sn, data, S)
        flow_result = compute_surrogate_flow(final_sn, data)
        dag_warnings = check_dag_safety(final_sn)
        print_report(final_sn, stats, flow_result, dag_warnings)

        sn_flow = compute_supernode_flow(final_sn, data)
        print_supernode_flow_report(sn_flow)

        # ── ADD THIS BLOCK ──────────────────────────────────────────────
        sn_flow_out = 'supernode_map_sn_flow.json'
        with open(sn_flow_out, 'w') as f:
            json.dump({
                'sn_names': sn_flow['sn_names'],
                'sn_adj': sn_flow['sn_adj'].tolist(),
                'F_sn': sn_flow['F_sn'].tolist(),
                'sn_reach': sn_flow['sn_reach'].tolist(),
                'sn_act_norm': sn_flow['sn_act_norm'].tolist(),
                'preservation': sn_flow['preservation'],
                'orig_reach_total': sn_flow['orig_reach_total'],
                'surr_reach_total': sn_flow['surr_reach_total'],
                'dominant_paths': sn_flow['dominant_paths'],
                'bottleneck_sns': sn_flow['bottleneck_sns'],
            }, f, indent=2)
        print(f'Supernode flow saved → {sn_flow_out}')
        # ────────────────────────────────────────────────────────────────

    print(f'\n  ➜  Use:  python auto_grouping.py --file {args.file} '
          f'--target-k {best_k} --max-layer-span {args.max_layer_span}')


if __name__ == '__main__':
    main()