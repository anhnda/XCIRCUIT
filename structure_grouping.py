"""
structure_grouping.py
─────────────────────────────────────────────────────────────────────────────
Automatically groups raw circuit nodes into supernodes by computing pairwise
node similarity from shared in/out neighbors (weighted by activation and
influence), then clustering.

Two clustering modes:
  --target-k K   Spectral clustering forcing exactly K middle supernodes.
                 Best when you know how many supernodes you want (e.g. < 10).
                 Uses the similarity matrix as a precomputed affinity.
  (default)      Hierarchical agglomerative clustering with a distance
                 threshold. Best for exploratory analysis via the dendrogram.

Pipeline:
  1. Load raw graph  (same as visualize_circuit.py)
  2. Build weighted adjacency A  (N×N)
  3. Compute similarity matrix S (N×N)
       S = 0.5 * cosine(A @ W @ A.T)   [shared out-neighbors]
         + 0.5 * cosine(A.T @ W @ A)   [shared in-neighbors]
       where W = diag(α·act_norm + β·inf_norm)
  4. Cluster middle nodes only (embedding + logit nodes kept fixed)
  5. Post-process: split clusters that span too many layers  →  DAG safety
  6. Verify no cycle risk between supernodes
  7. Output supernode map + summary statistics

Usage:
  # Spectral, force 7 supernodes (recommended for < 10 target):
  python structure_grouping.py --file subgraph/austin_plt.pt --target-k 7

  # Hierarchical with distance threshold (exploratory):
  python structure_grouping.py --file subgraph/austin_plt.pt \\
      --threshold 0.45 --max-layer-span 4 --alpha 0.5 --beta 0.5

  # dry-run with synthetic snapshot (no .pt file needed):
  python structure_grouping.py --synthetic --target-k 7
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────────────────
# 0.  HELPERS  (shared with visualize_circuit.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_layer(nid: str) -> int:
    if nid.startswith('E'):  return 0
    if nid.startswith('27'): return 27
    return int(nid.split('_')[0])


def parse_activation(attr_node: dict) -> float:
    a = attr_node.get('activation')
    if a is not None:
        return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100


def parse_influence(attr_node: dict) -> float:
    inf = attr_node.get('influence')
    return float(inf) if inf is not None else 0.0


def load_snapshot(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PREPARE GRAPH DATA  (mirrors visualize_circuit.py::prepare_graph_data)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_graph_data(raw: dict) -> dict:
    """
    Returns:
        kept_ids     : list[str]           node IDs in order
        adj          : torch.Tensor (N,N)  weighted adjacency (float32)
                       NOTE: logit column is NOT overwritten with influence.
                       adj reflects only real circuit connections (pruned_adj).
        act_values   : dict[str, float]    activation per node
        inf_values   : dict[str, float]    influence (→logit weight) per node
        inf_to_logit : dict[str, float]    direct influence score to logit
                       (node attribute — NOT injected into adj)
                       Use this for node sizing/coloring in visualizer,
                       NOT as an edge. Represents attribution strength,
                       not a real architectural connection.
        clerp        : dict[str, str]      human label per node
        layers       : list[int]           layer index per node
        node_inf     : torch.Tensor (N,)   normalised node influence scores
    """
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float().T   # ← .T added: receiver-indexed → sender-indexed
    attr      = raw['attr']
    logit_idx = len(kept_ids) - 1
    # ── REMOVED: influence injection into adj logit column ───────────────────
    # Previously: adj[i, logit_idx] = inf  ← mixed attribution into structure
    # Now: stored separately as inf_to_logit, rendered as node property
    # ─────────────────────────────────────────────────────────────────────────

    layers     = [parse_layer(n) for n in kept_ids]
    act_values = {n: parse_activation(attr[n]) for n in kept_ids}
    inf_values = {n: parse_influence(attr[n])  for n in kept_ids}
    clerp      = {n: attr[n].get('clerp', '')  for n in kept_ids}

    # Influence to logit: attribution score, separate from adj
    inf_to_logit = {
        nid: float(attr[nid].get('influence') or 0.0)
        for nid in kept_ids
    }

    # node_inf from raw if present, else fall back to influence_values
    if 'node_inf' in raw:
        node_inf = raw['node_inf'].float()
    else:
        ni = torch.tensor([inf_values[n] for n in kept_ids], dtype=torch.float32)
        node_inf = ni / (ni.max() + 1e-8)

    return dict(
        kept_ids     = kept_ids,
        adj          = adj,
        act_values   = act_values,
        inf_values   = inf_values,
        inf_to_logit = inf_to_logit,   # ← new, use for node sizing
        clerp        = clerp,
        layers       = layers,
        node_inf     = node_inf,
        logit_idx    = logit_idx,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMILARITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_norm(M: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine normalisation of a symmetric Gram matrix.
    M[i,j] = dot(profile_i, profile_j)  →  cos_sim[i,j]
    """
    diag = torch.sqrt(torch.diag(M).clamp(min=1e-8))
    return M / diag.unsqueeze(1) / diag.unsqueeze(0)


def compute_similarity(data: dict,
                        alpha: float = 0.5,
                        beta:  float = 0.5) -> torch.Tensor:
    """
    Compute pairwise node similarity.

    For each pair (u, v), similarity = how much they share the same
    in-neighbors AND out-neighbors, weighted by neighbor importance.

    Neighbor importance W[k] = α·act_norm[k] + β·inf_norm[k]

    S_out[i,j] = Σ_k  A[i,k] · W[k] · A[j,k]   (shared out-neighbors)
    S_in [i,j] = Σ_k  A[k,i] · W[k] · A[k,j]   (shared in-neighbors)
    S          = 0.5 · cosine(S_out) + 0.5 · cosine(S_in)

    Returns:
        S : torch.Tensor (N, N)  values in [0, 1]
    """
    kept_ids   = data['kept_ids']
    adj        = data['adj']                # (N, N)
    act_values = data['act_values']
    inf_values = data['inf_values']

    N = len(kept_ids)

    act_t = torch.tensor([act_values[n] for n in kept_ids], dtype=torch.float32)
    inf_t = torch.tensor([inf_values[n] for n in kept_ids], dtype=torch.float32)

    act_norm = act_t / (act_t.max() + 1e-8)
    inf_norm = inf_t / (inf_t.max() + 1e-8)

    importance = alpha * act_norm + beta * inf_norm   # (N,)
    W = torch.diag(importance)                         # (N, N)

    # Weighted Gram matrices
    S_out_raw = adj @ W @ adj.T      # (N,N): shared out-neighbors
    S_in_raw  = adj.T @ W @ adj      # (N,N): shared in-neighbors

    S_out_cos = _cosine_norm(S_out_raw)
    S_in_cos  = _cosine_norm(S_in_raw)

    S = 0.5 * S_out_cos + 0.5 * S_in_cos
    S = S.clamp(0.0, 1.0)

    return S


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

FIXED_TYPES = {'embedding', 'logit'}


def _is_fixed(nid: str) -> bool:
    return nid.startswith('E') or nid.startswith('27')


def cluster_middle_nodes(data: dict,
                          S: torch.Tensor,
                          threshold: float = 0.50,
                          linkage_method: str = 'average') -> dict:
    """
    Run hierarchical agglomerative clustering on the middle nodes only.

    Args:
        data           : prepared graph data
        S              : (N,N) similarity matrix
        threshold      : distance cut for fcluster  (distance = 1 - similarity)
        linkage_method : 'average' | 'ward' | 'complete' | 'single'
                         ward requires Euclidean — not recommended here.

    Returns:
        raw_clusters : dict  label(int) → list[str]  (middle nodes only)
        Z            : linkage matrix  (for dendrogram)
        middle_ids   : list[str]
        D_mid        : np.ndarray (M,M) distance matrix
    """
    kept_ids = data['kept_ids']

    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    M = len(middle_ids)

    S_mid = S[middle_idx][:, middle_idx].numpy()
    D_mid = (1.0 - S_mid).clip(0.0, 1.0)

    # scipy linkage needs condensed (upper-triangle) distance vector
    D_condensed = squareform(D_mid, checks=False)

    Z = linkage(D_condensed, method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion='distance')

    raw_clusters: dict[int, list[str]] = defaultdict(list)
    for nid, lbl in zip(middle_ids, labels):
        raw_clusters[int(lbl)].append(nid)

    return dict(
        raw_clusters = dict(raw_clusters),
        Z            = Z,
        middle_ids   = middle_ids,
        D_mid        = D_mid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3b. SPECTRAL CLUSTERING WITH FIXED TARGET-K
# ─────────────────────────────────────────────────────────────────────────────
def cluster_with_target_k(data: dict,
                           S: torch.Tensor,
                           target_k: int = 7,
                           max_layer_span: int = 4) -> dict:
    from sklearn.cluster import SpectralClustering

    kept_ids   = data['kept_ids']
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    M          = len(middle_ids)

    if target_k >= M:
        raise ValueError(f'target_k={target_k} >= number of middle nodes={M}. '
                         f'Choose a smaller target.')

    # Extract and symmetrise the middle-node submatrix
    S_mid = S[middle_idx][:, middle_idx].numpy()
    S_mid = ((S_mid + S_mid.T) / 2).clip(0.0, 1.0)

    # Similarity distribution
    upper = S_mid[np.triu_indices(M, k=1)]
    global_mean = float(upper.mean())
    print(f'  Similarity stats (middle nodes, upper triangle):')
    print(f'    mean={global_mean:.4f}  median={np.median(upper):.4f}  '
          f'p75={np.percentile(upper,75):.4f}  max={upper.max():.4f}')

    # ── Outlier detection ────────────────────────────────────────────────────
    max_sim_per_node  = S_mid.max(axis=1)
    outlier_threshold = global_mean *1.0
    outlier_mask      = max_sim_per_node < outlier_threshold
    core_mask         = ~outlier_mask

    outlier_ids = [middle_ids[i] for i in range(M) if outlier_mask[i]]
    core_ids    = [middle_ids[i] for i in range(M) if core_mask[i]]
    core_local  = [i for i in range(M) if core_mask[i]]   # indices into S_mid

    if outlier_ids:
        print(f'  Outliers ({len(outlier_ids)} nodes, max_sim < {outlier_threshold:.3f}):')
        for nid in outlier_ids:
            print(f'    {nid}')

    # ── Cluster only core nodes ──────────────────────────────────────────────
    S_core = S_mid[np.ix_(core_local, core_local)]
    n_core = len(core_ids)

    effective_k = min(target_k, n_core - 1)
    if effective_k < 2:
        # Degenerate: just one cluster for core
        raw_clusters = {0: core_ids}
    else:
        sc = SpectralClustering(
            n_clusters    = effective_k,
            affinity      = 'precomputed',
            assign_labels = 'kmeans',
            random_state  = 42,
            n_init        = 20,
        )
        labels = sc.fit_predict(S_core)
        raw_clusters: dict[int, list[str]] = defaultdict(list)
        for nid, lbl in zip(core_ids, labels):
            raw_clusters[int(lbl)].append(nid)

    print(f'  Spectral clusters (before DAG enforcement): {len(raw_clusters)}')

    # ── Assign each outlier to nearest-layer supernode ───────────────────────
    # Build a layer map for core clusters first, then assign outliers
    layers = {nid: parse_layer(nid) for nid in middle_ids}

    if outlier_ids:
        # Compute mean layer of each raw cluster
        cluster_mean_layer = {
            lbl: np.mean([layers[n] for n in members])
            for lbl, members in raw_clusters.items()
        }
        for nid in outlier_ids:
            nid_layer = layers[nid]
            nearest_lbl = min(cluster_mean_layer,
                              key=lambda lbl: abs(cluster_mean_layer[lbl] - nid_layer))
            raw_clusters[nearest_lbl].append(nid)
            print(f'  Outlier {nid} (L{nid_layer}) → cluster {nearest_lbl} '
                  f'(mean_layer={cluster_mean_layer[nearest_lbl]:.1f})')

    return enforce_dag(dict(raw_clusters), data, max_layer_span)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  POST-PROCESS: enforce DAG ordering
# ─────────────────────────────────────────────────────────────────────────────
def enforce_dag(raw_clusters: dict,
                data: dict,
                max_layer_span: int = 4) -> dict:
    layers = {nid: parse_layer(nid) for nid in
              [n for members in raw_clusters.values() for n in members]}

    queue = list(raw_clusters.values())
    final: list[list[str]] = []

    while queue:
        members = queue.pop()
        lvals   = sorted(set(layers[n] for n in members))
        span    = lvals[-1] - lvals[0]

        if span <= max_layer_span or len(members) == 1:
            final.append(members)
        else:
            # Split at the MIDPOINT of the layer range, not median of unique layers
            mid_layer = (lvals[0] + lvals[-1]) / 2
            lower = [n for n in members if layers[n] <= mid_layer]
            upper = [n for n in members if layers[n] >  mid_layer]

            # Guard: if split doesn't reduce either side, force split by membership
            if not lower or not upper:
                half = len(members) // 2
                lower = members[:half]
                upper = members[half:]

            if lower: queue.append(lower)
            if upper: queue.append(upper)

    # Sort supernodes by their minimum layer (topological order)
    final.sort(key=lambda m: min(layers[n] for n in m))
    # Post-hoc interleave resolution: split any cluster whose layer range
    # overlaps (interleaves) with another. Repeat until stable.
    changed = True
    while changed:
        changed = False
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                lo_i = min(layers[n] for n in final[i])
                hi_i = max(layers[n] for n in final[i])
                lo_j = min(layers[n] for n in final[j])
                hi_j = max(layers[n] for n in final[j])
                # Interleave: lo_i < lo_j < hi_i  OR  lo_j < lo_i < hi_j
                if (lo_i < lo_j < hi_i) or (lo_j < lo_i < hi_j):
                    # Split the cluster with the larger span
                    if (hi_i - lo_i) >= (hi_j - lo_j):
                        victim, other = i, j
                    else:
                        victim, other = j, i
                    lo_v = min(layers[n] for n in final[victim])
                    hi_v = max(layers[n] for n in final[victim])
                    # Split victim at the layer of the other cluster's lower bound
                    split_layer = min(layers[n] for n in final[other])
                    lower = [n for n in final[victim] if layers[n] < split_layer]
                    upper = [n for n in final[victim] if layers[n] >= split_layer]
                    if lower and upper:
                        final[victim] = lower
                        final.append(upper)
                        final.sort(key=lambda m: min(layers[n] for n in m))
                        changed = True
                        break  # restart outer loop
            if changed:
                break
    final_supernodes = {}
    for idx, members in enumerate(final):
        lo = min(layers[n] for n in members)
        hi = max(layers[n] for n in members)
        name = f'SN_{idx:02d}_L{lo}' if lo == hi else f'SN_{idx:02d}_L{lo}_{hi}'
        final_supernodes[name] = members

    # Add fixed nodes back
    fixed_emb   = [n for n in data['kept_ids'] if n.startswith('E')]
    fixed_logit = [n for n in data['kept_ids'] if n.startswith('27')]
    if fixed_emb:
        final_supernodes['SN_EMB']   = fixed_emb
    if fixed_logit:
        final_supernodes['SN_LOGIT'] = fixed_logit

    return final_supernodes


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DAG CYCLE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_dag_safety(final_supernodes: dict) -> list[tuple[str, str]]:
    """
    Flag supernode pairs whose layer ranges interleave:
        lo_a < lo_b < hi_a < hi_b  →  potential cycle A↔B
    Returns list of (sn_a, sn_b) warning pairs.
    """
    def layer_range(members):
        lvals = [parse_layer(n) for n in members]
        return min(lvals), max(lvals)

    ranges = {sn: layer_range(m) for sn, m in final_supernodes.items()}
    warnings = []
    sn_list  = list(ranges.keys())

    for i, sn_a in enumerate(sn_list):
        for sn_b in sn_list[i+1:]:
            lo_a, hi_a = ranges[sn_a]
            lo_b, hi_b = ranges[sn_b]
            # Check interleaving in both directions
            if lo_a < lo_b < hi_a < hi_b:
                warnings.append((sn_a, sn_b))
            elif lo_b < lo_a < hi_b < hi_a:
                warnings.append((sn_b, sn_a))

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_grouping(final_supernodes: dict,
                      data: dict,
                      S: torch.Tensor) -> dict:
    """
    For each supernode compute:
      - intra_sim_mean / min  : how similar are members to each other?
      - act_sum               : total activation carried
      - inf_sum               : total direct logit influence carried
      - layer_span            : hi_layer - lo_layer
    """
    kept_ids   = data['kept_ids']
    act_values = data['act_values']
    inf_values = data['inf_values']
    clerp      = data['clerp']

    id2idx = {nid: i for i, nid in enumerate(kept_ids)}

    stats = {}
    for sn, members in final_supernodes.items():
        idx   = [id2idx[n] for n in members if n in id2idx]
        lvals = [parse_layer(n) for n in members]

        # Intra-cluster similarity
        pairs = [(i, j) for ii, i in enumerate(idx) for j in idx[ii+1:]]
        if pairs:
            sims = [S[i, j].item() for i, j in pairs]
            intra_mean = float(np.mean(sims))
            intra_min  = float(np.min(sims))
        else:
            intra_mean = intra_min = 1.0  # single-node cluster

        stats[sn] = dict(
            n          = len(members),
            layer_lo   = min(lvals),
            layer_hi   = max(lvals),
            layer_span = max(lvals) - min(lvals),
            act_sum    = sum(act_values.get(n, 0) for n in members),
            inf_sum    = sum(inf_values.get(n, 0) for n in members),
            intra_sim_mean = intra_mean,
            intra_sim_min  = intra_min,
            members    = members,
            clerps     = [clerp.get(n, '') for n in members],
        )

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SURROGATE FLOW VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 7.  FLOW MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_to_logit(data: dict, max_hops: int = 6) -> dict:
    """
    Compute total multi-hop flow from each node to the logit node,
    following real edges in pruned_adj only (no influence injection).

    Flow semantics:
        F[i,j]       = act_norm[i] * adj[i,j]   (flow i sends to j)
        1-hop reach  = F[:, logit]
        2-hop reach  = (F @ F)[:, logit]         act[k] re-weighted via F[k,j]
        total        = Σ_{hop=1..max_hops} (F^hop)[:, logit]

    Why F^n works:
        (F@F)[i,j] = Σ_k F[i,k]*F[k,j]
                   = Σ_k act[i]*adj[i,k] * act[k]*adj[k,j]
        Each hop naturally re-weights by intermediate node activation. ✓

    Returns:
        dict with:
            reach        : torch.Tensor (N,)  total flow each node sends to logit
            reach_by_hop : list[tensor]       per-hop breakdown
            node_map     : dict nid → float   reach value per node
            total        : float              sum of all reach (= orig_to_logit)
    """
    kept_ids   = data['kept_ids']
    adj        = data['adj']
    act_values = data['act_values']
    logit_idx  = data['logit_idx']
    N          = len(kept_ids)

    act = torch.tensor(
        [act_values[n] for n in kept_ids],
        dtype=torch.float32
    )

    # Normalise activation to prevent numerical overflow across hops
    # (activations can be up to ~116, and F^6 would explode without this)
    act_norm = act / (act.max() + 1e-8)

    # F[i,j] = act_norm[i] * adj[i,j]
    F = act_norm.unsqueeze(1) * adj   # (N, N)

    reach        = torch.zeros(N)
    reach_by_hop = []
    current      = F.clone()

    for hop in range(max_hops):
        hop_reach = current[:, logit_idx]
        reach    += hop_reach
        reach_by_hop.append(hop_reach.clone())

        if hop < max_hops - 1:
            current = current @ F   # propagate one more hop

    node_map = {nid: reach[i].item() for i, nid in enumerate(kept_ids)}

    return dict(
        reach        = reach,
        reach_by_hop = reach_by_hop,
        node_map     = node_map,
        total        = reach.sum().item(),
    )


def compute_surrogate_flow(final_supernodes: dict, data: dict,
                           max_hops: int = 6) -> dict:
    """
    Two-part flow measurement:

    STRUCTURAL FLOW (multi-hop, real edges only):
        Uses compute_flow_to_logit() — traces actual circuit paths.
        reach[i] = total normalised flow node i sends to logit across all paths.
        Aggregated per supernode.

    ATTRIBUTION SUMMARY (causal, precomputed):
        Uses inf_to_logit — the influence score from circuit extraction.
        Represents total causal effect of each node on the logit,
        computed externally (e.g. activation patching).
        Aggregated per supernode.

    Both are reported separately in print_report.
    """
    kept_ids   = data['kept_ids']
    logit_idx  = data['logit_idx']
    logit_id   = kept_ids[logit_idx]
    inf_to_logit = data.get('inf_to_logit', {})

    # ── Structural flow ───────────────────────────────────────────────────────
    flow_result  = compute_flow_to_logit(data, max_hops=max_hops)
    node_map     = flow_result['node_map']

    orig_to_logit = sum(
        v for nid, v in node_map.items() if nid != logit_id
    )

    sn_structural: dict[str, float] = {}
    for sn, members in final_supernodes.items():
        sn_structural[sn] = sum(
            node_map.get(n, 0.0) for n in members if n != logit_id
        )

    surr_to_logit = sum(
        v for sn, v in sn_structural.items() if sn != 'SN_LOGIT'
    )

    ratio = surr_to_logit / (orig_to_logit + 1e-8)

    # ── Attribution summary ───────────────────────────────────────────────────
    sn_attribution: dict[str, float] = {}
    for sn, members in final_supernodes.items():
        sn_attribution[sn] = sum(
            inf_to_logit.get(n, 0.0) for n in members
        )

    total_attr = sum(sn_attribution.values()) + 1e-8

    # ── Supernode-level structural flow matrix (SN_A → SN_B) ─────────────────
    adj        = data['adj']
    act_values = data['act_values']
    act        = torch.tensor(
        [act_values[n] for n in kept_ids], dtype=torch.float32
    )
    flow_matrix = act.unsqueeze(1) * adj   # direct flow matrix (not normalised)
    id2idx      = {nid: i for i, nid in enumerate(kept_ids)}
    sn_idx      = {sn: [id2idx[n] for n in members if n in id2idx]
                   for sn, members in final_supernodes.items()}
    sn_names    = list(final_supernodes.keys())

    sn_flow: dict[str, dict[str, float]] = {}
    for src in sn_names:
        sn_flow[src] = {}
        for tgt in sn_names:
            if src == tgt:
                sn_flow[src][tgt] = 0.0
                continue
            sn_flow[src][tgt] = sum(
                flow_matrix[i, j].item()
                for i in sn_idx[src]
                for j in sn_idx[tgt]
            )

    return dict(
        # structural
        orig_to_logit  = orig_to_logit,
        surr_to_logit  = surr_to_logit,
        ratio          = ratio,
        sn_structural  = sn_structural,   # per-SN multi-hop reach
        sn_flow        = sn_flow,         # direct SN→SN flow matrix
        node_map       = node_map,        # per-node reach
        # attribution
        sn_attribution = sn_attribution,  # per-SN inf_to_logit sum
        total_attr     = total_attr,
    )
# ─────────────────────────────────────────────────────────────────────────────
# 8.  PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(final_supernodes: dict,
                 stats: dict,
                 flow_result: dict,
                 dag_warnings: list):

    SEP = '─' * 72

    print(f'\n{"═"*72}')
    print('  STRUCTURE GROUPING REPORT')
    print(f'{"═"*72}')

    print(f'\n  Total supernodes  : {len(final_supernodes)}')
    print(f'  Fixed (kept)      : SN_EMB, SN_LOGIT')
    print(f'  Clustered         : {len(final_supernodes) - 2}')

    print(f'\n{SEP}')
    print('  SUPERNODE SUMMARY')
    print(SEP)
    header = f"  {'Name':<22} {'n':>3}  {'layers':>10}  {'act_sum':>8}  " \
             f"{'inf_sum':>8}  {'intra_sim':>10}"
    print(header)
    print(f'  {"-"*68}')
    for sn, st in stats.items():
        layer_str = (f"L{st['layer_lo']}"
                     if st['layer_span'] == 0
                     else f"L{st['layer_lo']}–{st['layer_hi']}")
        print(f"  {sn:<22} {st['n']:>3}  {layer_str:>10}  "
              f"{st['act_sum']:>8.2f}  {st['inf_sum']:>8.4f}  "
              f"{st['intra_sim_mean']:>10.4f}")

    print(f'\n{SEP}')
    print('  MEMBER DETAILS')
    print(SEP)
    for sn, st in stats.items():
        print(f'\n  ┌── {sn}  '
              f'(n={st["n"]}, layers L{st["layer_lo"]}–{st["layer_hi"]}, '
              f'intra_sim_min={st["intra_sim_min"]:.3f})')
        for nid, c in zip(st['members'], st['clerps']):
            lyr = parse_layer(nid)
            print(f'  │   {nid:<20}  L{lyr:<3}  {c}')
        print(f'  └{"─"*60}')

    print(f'\n{SEP}')
    print('  FLOW PRESERVATION  (multi-hop structural, real edges only)')
    print(SEP)
    fr = flow_result
    print(f"  Original  reach → logit : {fr['orig_to_logit']:.4f}  (normalised)")
    print(f"  Surrogate reach → logit : {fr['surr_to_logit']:.4f}")
    print(f"  Preservation ratio      : {fr['ratio']:.4f}")
    # Replace the three-branch if/elif/else on ratio with:
    if fr['orig_to_logit'] < 1e-6:
        print('  [INFO] No structural edges to logit in adj — attribution-only graph.')
        print('         Flow ratio check skipped. See ATTRIBUTION SUMMARY below.')
    elif 0.8 <= fr['ratio'] <= 1.2:
        print('  [PASS] Within 20% tolerance.')
    elif 0.5 <= fr['ratio'] < 0.8:
        print(f"  [WARN] Surrogate captures {fr['ratio'] * 100:.1f}% of total reach.")
    else:
        print(f"  [FAIL] Ratio {fr['ratio']:.3f} — supernode graph may be too coarse.")

    print(f'\n  Structural reach → logit (by supernode):')
    total_struct = fr['surr_to_logit'] + 1e-8
    for sn, val in fr['sn_structural'].items():
        if sn == 'SN_LOGIT': continue
        pct = val / total_struct * 100
        bar = '█' * int(pct / 2)
        print(f"    {sn:<22} {val:>8.4f}  ({pct:>5.1f}%)  {bar}")

    print(f'\n{SEP}')
    print('  ATTRIBUTION SUMMARY  (causal influence scores, precomputed)')
    print(SEP)
    print(f'  (Captures total causal effect per supernode on logit,')
    print(f'   independent of path — NOT an edge, a node property.)\n')
    total_attr = fr['total_attr']
    for sn, val in fr['sn_attribution'].items():
        if sn == 'SN_LOGIT': continue
        pct = val / total_attr * 100
        bar = '█' * int(pct / 2)
        print(f"    {sn:<22} {val:>8.4f}  ({pct:>5.1f}%)  {bar}")

    print(f'\n{SEP}')
    print('  DAG SAFETY CHECK')
    print(SEP)
    if dag_warnings:
        for sn_a, sn_b in dag_warnings:
            print(f'  [WARN] Potential cycle: {sn_a} ↔ {sn_b}')
    else:
        print('  [PASS] No interleaving layer ranges detected.')

    print(f'\n{"═"*72}\n')


# ─────────────────────────────────────────────────────────────────────────────
# 9.  OPTIONAL: dendrogram (requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z, middle_ids: list, threshold: float,
                    out_path: str = 'dendrogram.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [INFO] matplotlib not available — skipping dendrogram.')
        return

    fig, ax = plt.subplots(figsize=(max(14, len(middle_ids) * 0.22), 5))
    dendrogram(Z, labels=middle_ids, leaf_rotation=90,
               leaf_font_size=6, ax=ax, color_threshold=threshold)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=0.8,
               label=f'threshold={threshold}')
    ax.set_title('Hierarchical Clustering — Middle Nodes', fontsize=11)
    ax.set_ylabel('Distance  (1 − similarity)', fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Dendrogram saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SYNTHETIC SNAPSHOT  (fallback when no .pt file is available)
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_snapshot() -> dict:
    """Reconstruct the austin_plt snapshot from hard-coded values."""
    kept_ids = [
        '16_25_9', '16_12678_9', '16_4298_10', '16_13497_10',
        '17_7178_10',
        '18_1026_10', '18_1437_10', '18_3852_10',
        '18_5495_10', '18_6101_10', '18_8959_10', '18_16041_10',
        '19_7477_9', '19_37_10', '19_1445_10', '19_2439_10',
        '19_2695_10', '19_7477_10',
        '20_15589_9', '20_114_10', '20_5916_10', '20_6026_10',
        '20_7507_10', '20_15276_10', '20_15589_10',
        '21_5943_10', '21_6316_10', '21_6795_10', '21_14975_10',
        '22_31_10', '22_3064_10', '22_3551_10', '22_4999_10',
        '22_11718_10',
        '23_2288_10', '23_6617_10', '23_11444_10', '23_12237_10',
        '23_12918_10', '23_13193_10', '23_13541_10', '23_13841_10',
        '23_15366_10',
        '24_709_10', '24_6044_10', '24_6394_10', '24_13277_10',
        '24_15013_10', '24_15627_10', '24_15694_10', '24_16258_10',
        '25_553_10', '25_583_10', '25_762_10', '25_2687_10',
        '25_4259_10', '25_4679_10', '25_4717_10', '25_4886_10',
        '25_13300_10', '25_16302_10',
        'E_6037_4', 'E_2329_7', 'E_26865_9',
        '27_22605_10',
    ]
    act_v = {
        '16_25_9':28.16,'16_12678_9':32.06,'16_4298_10':19.93,'16_13497_10':10.26,
        '17_7178_10':27.50,'18_1026_10':15.99,'18_1437_10':9.26,'18_3852_10':8.45,
        '18_5495_10':12.45,'18_6101_10':20.99,'18_8959_10':39.96,'18_16041_10':9.47,
        '19_7477_9':55.78,'19_37_10':9.70,'19_1445_10':35.45,'19_2439_10':17.40,
        '19_2695_10':17.39,'19_7477_10':26.97,'20_15589_9':45.66,'20_114_10':17.25,
        '20_5916_10':53.43,'20_6026_10':17.02,'20_7507_10':8.21,'20_15276_10':9.98,
        '20_15589_10':49.56,'21_5943_10':56.37,'21_6316_10':10.86,'21_6795_10':16.33,
        '21_14975_10':12.31,'22_31_10':24.70,'22_3064_10':58.50,'22_3551_10':48.36,
        '22_4999_10':37.47,'22_11718_10':34.40,'23_2288_10':14.86,'23_6617_10':9.10,
        '23_11444_10':31.55,'23_12237_10':54.44,'23_12918_10':18.48,'23_13193_10':26.53,
        '23_13541_10':13.38,'23_13841_10':23.95,'23_15366_10':23.05,'24_709_10':10.41,
        '24_6044_10':55.07,'24_6394_10':25.96,'24_13277_10':116.57,'24_15013_10':29.57,
        '24_15627_10':26.43,'24_15694_10':15.49,'24_16258_10':22.16,'25_553_10':51.21,
        '25_583_10':25.03,'25_762_10':11.22,'25_2687_10':10.40,'25_4259_10':29.21,
        '25_4679_10':16.61,'25_4717_10':31.33,'25_4886_10':27.92,'25_13300_10':22.51,
        '25_16302_10':39.76,'E_6037_4':4.15,'E_2329_7':22.77,'E_26865_9':17.14,
        '27_22605_10':45.04,
    }
    inf_v = {
        '16_25_9':0.3085,'16_12678_9':0.4794,'16_4298_10':0.4371,'16_13497_10':0.5397,
        '17_7178_10':0.4327,'18_1026_10':0.4726,'18_1437_10':0.3878,'18_3852_10':0.5543,
        '18_5495_10':0.3778,'18_6101_10':0.3535,'18_8959_10':0.3443,'18_16041_10':0.5711,
        '19_7477_9':0.4462,'19_37_10':0.4213,'19_1445_10':0.3703,'19_2439_10':0.4290,
        '19_2695_10':0.3482,'19_7477_10':0.3802,'20_15589_9':0.3150,'20_114_10':0.5014,
        '20_5916_10':0.4005,'20_6026_10':0.5161,'20_7507_10':0.4665,'20_15276_10':0.4912,
        '20_15589_10':0.2618,'21_5943_10':0.2895,'21_6316_10':0.5390,'21_6795_10':0.3928,
        '21_14975_10':0.4778,'22_31_10':0.4495,'22_3064_10':0.3551,'22_3551_10':0.3996,
        '22_4999_10':0.3403,'22_11718_10':0.3581,'23_2288_10':0.4864,'23_6617_10':0.4076,
        '23_11444_10':0.3977,'23_12237_10':0.2938,'23_12918_10':0.4414,'23_13193_10':0.3919,
        '23_13541_10':0.4482,'23_13841_10':0.4110,'23_15366_10':0.4101,'24_709_10':0.5278,
        '24_6044_10':0.2694,'24_6394_10':0.3181,'24_13277_10':0.3207,'24_15013_10':0.3967,
        '24_15627_10':0.5335,'24_15694_10':0.3857,'24_16258_10':0.4341,'25_553_10':0.3909,
        '25_583_10':0.4400,'25_762_10':0.5573,'25_2687_10':0.5372,'25_4259_10':0.3813,
        '25_4679_10':0.4768,'25_4717_10':0.5019,'25_4886_10':0.4710,'25_13300_10':0.3118,
        '25_16302_10':0.4198,'E_6037_4':0.0415,'E_2329_7':0.2277,'E_26865_9':0.1714,
        '27_22605_10':None,
    }
    clerp = {
        '16_25_9':'Texas legal documents','16_12678_9':'cities',
        '16_4298_10':'capital','16_13497_10':'Numbers and parameters',
        '17_7178_10':'government buildings','18_1026_10':'country names',
        '18_1437_10':'Legal documents from Texas','18_3852_10':'Locations',
        '18_5495_10':'locations','18_6101_10':'capital cities',
        '18_8959_10':'government/state','18_16041_10':'capital',
        '19_7477_9':'Dallas','19_37_10':'Places','19_1445_10':'Downtowns of cities',
        '19_2439_10':'Politics and government','19_2695_10':'cities',
        '19_7477_10':'Dallas','20_15589_9':'Texas','20_114_10':'Oklahoma locations',
        '20_5916_10':'capital','20_6026_10':'political titles',
        '20_7507_10':'countries','20_15276_10':'Dallas sports',
        '20_15589_10':'Texas','21_5943_10':'cities','21_6316_10':'special',
        '21_6795_10':'geographic place names near Texas',
        '21_14975_10':'state/states','22_31_10':'government and policy',
        '22_3064_10':'Texas/Dallas','22_3551_10':'Place names and legal cases',
        '22_4999_10':'Locations','22_11718_10':'Texas locations',
        '23_2288_10':'Texas','23_6617_10':'Locations',
        '23_11444_10':'cities and places','23_12237_10':'Cities and states names',
        '23_12918_10':'Texas','23_13193_10':'Legal and Southern place names',
        '23_13541_10':'News articles','23_13841_10':'towns and cities',
        '23_15366_10':'Code snippets','24_709_10':'patent identifiers',
        '24_6044_10':'in','24_6394_10':'locations',
        '24_13277_10':'Romance languages','24_15013_10':'in',
        '24_15627_10':'Locations','24_15694_10':'US states',
        '24_16258_10':'Detects place names','25_553_10':'general English text',
        '25_583_10':'city names','25_762_10':'international locations',
        '25_2687_10':'locations','25_4259_10':'place names',
        '25_4679_10':'locations','25_4717_10':'unusual/fantastical narratives',
        '25_4886_10':'last names','25_13300_10':'Texas',
        '25_16302_10':'Legal/court cases','E_6037_4':'Emb: " capital"',
        'E_2329_7':'Emb: " state"','E_26865_9':'Emb: " Dallas"',
        '27_22605_10':'Output " Austin" (p=0.450)',
    }

    N         = len(kept_ids)
    adj       = torch.zeros(N, N)
    logit_idx = N - 1
    id2i      = {nid: i for i, nid in enumerate(kept_ids)}

    # Real influence weights → logit column
    # for nid, inf in inf_v.items():
    #     if inf is not None:
    #         adj[id2i[nid], logit_idx] = inf

    # Synthetic inter-layer edges
    def get_layer(nid):
        if nid.startswith('E'): return 0
        if nid.startswith('27'): return 27
        return int(nid.split('_')[0])

    layers = [get_layer(n) for n in kept_ids]
    torch.manual_seed(42)
    for i in range(N):
        for j in range(N):
            if i == j or j == logit_idx: continue
            if layers[i] < layers[j]:
                w = (act_v[kept_ids[i]] / 200.0) * torch.rand(1).item()
                if w > 0.05:
                    adj[i, j] = round(w, 4)

    # Build attr dict
    attr = {}
    for nid in kept_ids:
        is_emb = nid.startswith('E')
        is_log = nid.startswith('27')
        attr[nid] = dict(
            activation     = act_v[nid] if not (is_emb or is_log) else None,
            influence      = inf_v.get(nid),
            clerp          = clerp[nid],
            is_target_logit= is_log,
            token_prob     = 0.4504 if is_log else None,
            ctx_idx        = int(nid.split('_')[2]) if '_' in nid and not is_emb else 0,
        )

    return dict(kept_ids=kept_ids, pruned_adj=adj, attr=attr)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automatic supernode grouping for circuit graphs')
    parser.add_argument('--file',           type=str,   default='subgraph/austin_plt.pt')
    parser.add_argument('--synthetic',      action='store_true',
                        help='Use built-in synthetic snapshot (no .pt file needed)')
    parser.add_argument('--target-k',       type=int,   default=None,
                        help='Force exactly K middle supernodes via spectral '
                             'clustering (recommended when you want < 10 total). '
                             'Overrides --threshold and --linkage.')
    parser.add_argument('--threshold',      type=float, default=0.50,
                        help='Distance threshold for hierarchical clustering '
                             '(used only when --target-k is not set, default 0.50)')
    parser.add_argument('--linkage',        type=str,   default='average',
                        choices=['average', 'complete', 'single'],
                        help='Linkage method for hierarchical clustering (default: average)')
    parser.add_argument('--max-layer-span', type=int,   default=4,
                        help='Max layer range allowed in one supernode (default 4)')
    parser.add_argument('--alpha',          type=float, default=0.5,
                        help='Weight for activation in neighbor importance (default 0.5)')
    parser.add_argument('--beta',           type=float, default=0.5,
                        help='Weight for influence in neighbor importance (default 0.5)')
    parser.add_argument('--dendrogram',     type=str,   default='dendrogram.png',
                        help='Output path for dendrogram image (hierarchical mode only)')
    parser.add_argument('--out-json',       type=str,   default='supernode_map.json',
                        help='Output path for supernode map JSON')
    args = parser.parse_args()

    # ── Load data
    if args.synthetic:
        print('Using synthetic snapshot...')
        raw = build_synthetic_snapshot()
    else:
        print(f'Loading {args.file}...')
        raw = load_snapshot(args.file)

    data = prepare_graph_data(raw)
    N    = len(data['kept_ids'])
    print(f'  Nodes     : {N}')
    print(f'  Adj shape : {data["adj"].shape}')

    # ── Similarity
    print('\nComputing similarity matrix...')
    S = compute_similarity(data, alpha=args.alpha, beta=args.beta)
    print(f'  S shape   : {S.shape}')
    print(f'  S range   : [{S.min():.4f}, {S.max():.4f}]')

    # ── Cluster
    if args.target_k is not None:
        print(f'\nSpectral clustering (target_k={args.target_k}, '
              f'max_layer_span={args.max_layer_span})...')
        final_supernodes = cluster_with_target_k(
            data, S,
            target_k       = args.target_k,
            max_layer_span = args.max_layer_span,
        )
        print(f'  Final supernodes : {len(final_supernodes)}')
    else:
        print(f'\nHierarchical clustering '
              f'(threshold={args.threshold}, linkage={args.linkage})...')
        cluster_result = cluster_middle_nodes(
            data, S,
            threshold      = args.threshold,
            linkage_method = args.linkage,
        )
        n_raw = len(cluster_result['raw_clusters'])
        print(f'  Raw clusters (before DAG enforcement) : {n_raw}')

        # Dendrogram only available in hierarchical mode
        plot_dendrogram(
            cluster_result['Z'],
            cluster_result['middle_ids'],
            threshold = args.threshold,
            out_path  = args.dendrogram,
        )

        print(f'\nEnforcing DAG (max_layer_span={args.max_layer_span})...')
        final_supernodes = enforce_dag(
            cluster_result['raw_clusters'],
            data,
            max_layer_span = args.max_layer_span,
        )
        print(f'  Final supernodes : {len(final_supernodes)}')

    # ── DAG check
    dag_warnings = check_dag_safety(final_supernodes)

    # ── Quality stats
    stats = evaluate_grouping(final_supernodes, data, S)

    # ── Flow verification
    flow_result = compute_surrogate_flow(final_supernodes, data)

    # ── Report
    print_report(final_supernodes, stats, flow_result, dag_warnings)

    # ── Save JSON
    out_map = {sn: members for sn, members in final_supernodes.items()}
    with open(args.out_json, 'w') as f:
        json.dump(out_map, f, indent=2)
    print(f'Supernode map saved → {args.out_json}')


if __name__ == '__main__':
    main()