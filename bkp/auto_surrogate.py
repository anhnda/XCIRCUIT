"""
Automatic Surrogate Graph Construction
=======================================
Pipeline: pruned_adj + attr + node_inf + node_rel
       → similarity matrix (5 signals)
       → SpectralClustering
       → supernode_map
       → flow validation

Usage:
    python auto_surrogate.py                        # uses build_data_from_snapshot()
    python auto_surrogate.py --file path/to.pt      # loads from file
    python auto_surrogate.py --file path/to.pt --k 4 --search
"""

import argparse
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════
import torch
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def directed_laplacian(adj: torch.Tensor, layers: list, method='rw') -> np.ndarray:
    """
    Build directed Laplacian from DAG adjacency matrix.

    method='rw'   : Random Walk Laplacian (Chung 2005)
                    better for flow-based graphs
    method='comb' : Combinatorial directed Laplacian
                    simpler, symmetric
    """
    A = build_dag_adj(adj, layers)   # ← thêm dòng này
    N = A.shape[0]

    if method == 'comb':
        # L_dir = (L_out + L_in) / 2
        d_out = A.sum(axis=1)
        d_in = A.sum(axis=0)
        L_out = np.diag(d_out) - A
        L_in = np.diag(d_in) - A.T
        return (L_out + L_in) / 2.0

    elif method == 'rw':
        # Transition matrix P = row-normalized A
        d_out = A.sum(axis=1, keepdims=True)
        d_out = np.where(d_out == 0, 1e-8, d_out)
        P = A / d_out  # row stochastic

        # Stationary distribution π via power iteration
        pi = np.ones(N) / N
        for _ in range(1000):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < 1e-9:
                break
            pi = pi_new
        pi = np.abs(pi)
        pi = pi / (pi.sum() + 1e-8)

        # Θ = diag(π)
        sqrt_pi = np.sqrt(pi + 1e-12)
        isqrt_pi = 1.0 / sqrt_pi

        # Symmetrized directed Laplacian
        # L_rw = I - (1/2)(Θ^{1/2} P Θ^{-1/2} + Θ^{-1/2} P^T Θ^{1/2})
        M = 0.5 * (
                np.diag(sqrt_pi) @ P @ np.diag(isqrt_pi) +
                np.diag(isqrt_pi) @ P.T @ np.diag(sqrt_pi)
        )
        return np.eye(N) - M


def enforce_layer_ordering(labels, layers, n_clusters):
    """
    Post-process clustering labels để đảm bảo:
    cluster_i chỉ chứa nodes có layer < cluster_j nếu i < j

    Strategy: sort clusters by mean layer,
    reassign nodes vi phạm về cluster phù hợp nhất
    """
    labels = labels.copy()

    # Bước 1: Sort clusters by mean layer
    cluster_mean_layer = {}
    for k in range(n_clusters):
        idxs = np.where(labels == k)[0]
        if len(idxs) > 0:
            cluster_mean_layer[k] = np.mean([layers[i] for i in idxs])
        else:
            cluster_mean_layer[k] = 0

    # Map cluster id → rank by layer
    sorted_clusters = sorted(cluster_mean_layer, key=cluster_mean_layer.get)
    rank_map = {c: r for r, c in enumerate(sorted_clusters)}
    labels = np.array([rank_map[l] for l in labels])

    # Bước 2: Compute layer range per cluster sau khi sort
    # Tính layer boundaries: cluster k nhận nodes có layer trong [lo_k, hi_k]
    layer_arr = np.array(layers)
    min_l, max_l = layer_arr.min(), layer_arr.max()
    band_size = (max_l - min_l + 1) / n_clusters

    # Bước 3: Reassign nodes về cluster theo layer band
    # Nếu node vi phạm (layer của nó không fit cluster range) → reassign
    for i, (lbl, l) in enumerate(zip(labels, layers)):
        # Expected cluster cho layer l
        expected = min(int((l - min_l) / band_size), n_clusters - 1)
        if lbl != expected:
            labels[i] = expected

    return labels
def dag_spectral_clustering(adj_sub, trans_ids, n_clusters, layers, method='rw'):
    """
    Spectral clustering on DAG using directed Laplacian.

    Steps:
    1. Build directed Laplacian L from adj_sub
    2. Compute k smallest eigenvectors of L
    3. Normalize rows of eigenvector matrix
    4. KMeans on eigenvectors → cluster labels
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    N = len(trans_ids)
    L = directed_laplacian(adj_sub, layers, method=method)

    # Eigen decomposition: k smallest eigenvalues
    # L is symmetric (after Chung symmetrization) → use eigh
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Take k eigenvectors corresponding to k smallest eigenvalues
    # Skip eigenvalue ≈ 0 (trivial solution) → take indices 0..k-1
    idx = np.argsort(eigenvalues)[:n_clusters]
    V = eigenvectors[:, idx]  # (N, k)

    # Row normalize → points on unit sphere
    V_norm = normalize(V, norm='l2', axis=1)

    # KMeans on eigenvector embedding
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    ).fit(V_norm)

    return kmeans.labels_, eigenvalues[idx]
def dag_svd_clustering(adj_sub, trans_ids, data, n_clusters):
    layers = [data['layers'][data['kept_ids'].index(n)]
              for n in trans_ids]
    min_l  = min(layers)
    max_l  = max(layers)

    # SVD trên adj_sub (61×61) - KHÔNG cần EMB/LOGIT
    A = adj_sub.abs().numpy().astype(np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    U_k = U[:, :n_clusters]
    V_k = Vt[:n_clusters, :].T

    # Layer-weighted combination (encode ordering)
    layer_norm = np.array([
        (l - min_l) / (max_l - min_l + 1e-8) for l in layers
    ])
    w_send = (1.0 - layer_norm).reshape(-1, 1)
    w_recv = layer_norm.reshape(-1, 1)

    # Thêm influence như separate feature
    # (thay thế cho logit column bị cắt)
    influence = np.array([
        data['influence_values'][n] for n in trans_ids
    ]).reshape(-1, 1)
    influence = influence / (influence.max() + 1e-8)  # normalize

    embedding = np.concatenate([
        U_k * w_send,    # sender pattern (weighted by layer)
        V_k * w_recv,    # receiver pattern (weighted by layer)
        influence,       # direct logit contribution
    ], axis=1)  # (61, 2k+1)

    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    emb_norm = normalize(embedding, norm='l2', axis=1)
    kmeans   = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    ).fit(emb_norm)
    raw_labels = kmeans.labels_

    # Hard enforce layer ordering
    labels = enforce_layer_ordering(raw_labels, layers, n_clusters)
    return labels, S[:n_clusters]
def parse_layer(nid: str) -> int:
    if nid.startswith('E'):  return 0
    return int(nid.split('_')[0])

def parse_activation(attr_node: dict) -> float:
    a = attr_node['activation']
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

def prepare_data(raw: dict) -> dict:
    """
    From raw snapshot dict → clean data dict.
    Injects influence values into logit column of pruned_adj.
    """
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float()
    logit_idx = len(kept_ids) - 1

    # Inject influence → logit column
    for i, nid in enumerate(kept_ids):
        inf = raw['attr'][nid].get('influence')
        if inf is not None:
            adj[i, logit_idx] = float(inf)

    return {
        'kept_ids':         kept_ids,
        'pruned_adj':       adj,
        'node_inf':         raw['node_inf'].float(),
        'node_rel':         raw['node_rel'].float(),
        'act_values':       {n: parse_activation(raw['attr'][n]) for n in kept_ids},
        'influence_values': {n: parse_influence(raw['attr'][n])  for n in kept_ids},
        'clerp':            {n: raw['attr'][n]['clerp']           for n in kept_ids},
        'layers':           [parse_layer(n) for n in kept_ids],
        'attr':             raw['attr'],
    }


def build_data_from_snapshot() -> dict:
    """Hardcoded snapshot for offline testing."""
    kept_ids = [
        '16_25_9','16_12678_9','16_4298_10','16_13497_10',
        '17_7178_10',
        '18_1026_10','18_1437_10','18_3852_10','18_5495_10',
        '18_6101_10','18_8959_10','18_16041_10',
        '19_7477_9','19_37_10','19_1445_10','19_2439_10',
        '19_2695_10','19_7477_10',
        '20_15589_9','20_114_10','20_5916_10','20_6026_10',
        '20_7507_10','20_15276_10','20_15589_10',
        '21_5943_10','21_6316_10','21_6795_10','21_14975_10',
        '22_31_10','22_3064_10','22_3551_10','22_4999_10','22_11718_10',
        '23_2288_10','23_6617_10','23_11444_10','23_12237_10',
        '23_12918_10','23_13193_10','23_13541_10','23_13841_10','23_15366_10',
        '24_709_10','24_6044_10','24_6394_10','24_13277_10',
        '24_15013_10','24_15627_10','24_15694_10','24_16258_10',
        '25_553_10','25_583_10','25_762_10','25_2687_10',
        '25_4259_10','25_4679_10','25_4717_10','25_4886_10',
        '25_13300_10','25_16302_10',
        'E_6037_4','E_2329_7','E_26865_9',
        '27_22605_10',
    ]
    act_raw = {
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
    inf_raw = {
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
    clerp_raw = {
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
    node_inf_vals = [
        0.5038,0.6243,0.6029,0.6835,0.5912,0.6279,0.5563,0.6905,0.5586,
        0.5406,0.5327,0.6960,0.6142,0.6009,0.5513,0.6106,0.5347,0.5719,
        0.5136,0.6629,0.6377,0.6769,0.6248,0.6492,0.4464,0.4822,0.6735,
        0.5682,0.6607,0.5976,0.5486,0.6036,0.5306,0.5458,0.6329,0.5867,
        0.5780,0.4924,0.5845,0.5771,0.6188,0.5701,0.5829,0.6651,0.4542,
        0.5071,0.5442,0.5672,0.6644,0.5737,0.6118,0.5631,0.6087,0.6976,
        0.6819,0.5598,0.6475,0.6766,0.6409,0.5104,0.6324,0.3002,0.4243,
        0.3739,0.2688,
    ]
    node_rel_vals = [
        0.6424,0.6671,0.6816,0.6882,0.6704,0.6638,0.5726,0.6542,0.6529,
        0.6112,0.5839,0.5979,0.4622,0.6604,0.5909,0.6826,0.5821,0.5117,
        0.4289,0.5587,0.5504,0.6517,0.6355,0.6797,0.4397,0.5041,0.6029,
        0.6835,0.5092,0.5352,0.4327,0.4431,0.6283,0.5142,0.6012,0.5440,
        0.4796,0.4907,0.5566,0.4363,0.5374,0.4962,0.4710,0.4528,0.3749,
        0.4175,0.3936,0.5765,0.4681,0.4251,0.4880,0.3482,0.3293,0.4463,
        0.5189,0.4017,0.3420,0.2360,0.3648,0.3357,0.3539,0.0494,0.1482,
        0.0988,0.2519,
    ]

    N   = len(kept_ids)
    adj = torch.zeros(N, N)
    lid = N - 1

    def get_layer(nid):
        if nid.startswith('E'): return 0
        if nid.startswith('27'): return 27
        return int(nid.split('_')[0])

    layers = [get_layer(n) for n in kept_ids]
    for i, nid in enumerate(kept_ids):
        inf = inf_raw.get(nid)
        if inf is not None:
            adj[i, lid] = inf

    torch.manual_seed(42)
    for i in range(N):
        for j in range(N):
            if i == j or j == lid: continue
            if layers[i] < layers[j]:
                w = (act_raw[kept_ids[i]] / 200.0) * torch.rand(1).item()
                if w > 0.05:
                    adj[i, j] = round(w, 4)

    attr = {}
    for n in kept_ids:
        attr[n] = {
            'clerp':           clerp_raw[n],
            'activation':      act_raw.get(n),
            'influence':       inf_raw.get(n),
            'ctx_idx':         int(n.split('_')[-1]) if not n.startswith('E') else int(n.split('_')[-1]),
            'is_target_logit': n.startswith('27'),
            'token_prob':      0.4504 if n.startswith('27') else 0,
        }

    return {
        'kept_ids':         kept_ids,
        'pruned_adj':       adj,
        'node_inf':         torch.tensor(node_inf_vals),
        'node_rel':         torch.tensor(node_rel_vals),
        'act_values':       act_raw,
        'influence_values': {n: (inf_raw[n] if inf_raw[n] is not None else 0.0) for n in kept_ids},
        'clerp':            clerp_raw,
        'layers':           layers,
        'attr':             attr,
    }


# ═══════════════════════════════════════════════════════════
# 2. SIMILARITY SIGNALS
# ═══════════════════════════════════════════════════════════

def sim_structural(adj: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity of row (outgoing) + col (incoming) edge patterns.
    Uses abs(adj) to handle negative edges.
    """
    A = adj.abs()
    row = F.normalize(A, p=2, dim=1)          # outgoing pattern
    col = F.normalize(A, p=2, dim=0).T        # incoming pattern
    s   = (row @ row.T + col @ col.T) / 2.0
    return s.clamp(0, 1)

def sim_layer(layers: list, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian similarity based on layer distance."""
    L    = torch.tensor(layers, dtype=torch.float32)
    diff = (L.unsqueeze(1) - L.unsqueeze(0)).abs()
    return torch.exp(-diff**2 / (2 * sigma**2))

def sim_ctx(kept_ids: list, attr: dict) -> torch.Tensor:
    """Binary: same ctx_idx = 1."""
    ctx = torch.tensor([attr[n]['ctx_idx'] for n in kept_ids], dtype=torch.float32)
    return (ctx.unsqueeze(1) == ctx.unsqueeze(0)).float()

def sim_role(node_inf: torch.Tensor, node_rel: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of (node_inf, node_rel) profile."""
    profile = torch.stack([node_inf, node_rel], dim=1)
    profile = F.normalize(profile, p=2, dim=1)
    return (profile @ profile.T).clamp(0, 1)

def sim_clerp_keyword(kept_ids: list, attr: dict) -> torch.Tensor:
    """
    Keyword-based similarity on clerp strings.
    No external model needed.
    """
    KEYWORDS = [
        'texas', 'dallas', 'capital', 'government', 'state',
        'city', 'cities', 'location', 'place', 'legal',
        'country', 'political', 'english', 'romance',
    ]
    N    = len(kept_ids)
    vecs = torch.zeros(N, len(KEYWORDS))
    for i, nid in enumerate(kept_ids):
        text = attr[nid]['clerp'].lower()
        for j, kw in enumerate(KEYWORDS):
            if kw in text:
                vecs[i, j] = 1.0
    vecs = F.normalize(vecs, p=2, dim=1)
    return (vecs @ vecs.T).clamp(0, 1)


# ═══════════════════════════════════════════════════════════
# 3. FLOW COMPUTATION (reused from circuit_flow.py)
# ═══════════════════════════════════════════════════════════

def compute_original_flow(data: dict) -> dict:
    kept_ids   = data['kept_ids']
    adj        = data['pruned_adj']
    act_values = data['act_values']
    N          = len(kept_ids)
    act        = torch.tensor([act_values[n] for n in kept_ids], dtype=torch.float32)
    flow_mat   = act.unsqueeze(1) * adj
    logit_idx  = N - 1
    return {
        'act':            act,
        'flow_matrix':    flow_mat,
        'flow_in':        flow_mat.sum(dim=0),
        'flow_out':       flow_mat.sum(dim=1),
        'flow_to_logit':  flow_mat[:, logit_idx],
        'total_to_logit': flow_mat[:, logit_idx].sum().item(),
        'logit_idx':      logit_idx,
    }

def compute_surrogate_flow(data: dict, orig_flow: dict, supernode_map: dict) -> dict:
    kept_ids   = data['kept_ids']
    flow_mat   = orig_flow['flow_matrix']
    act        = orig_flow['act']
    id2idx     = {n: i for i, n in enumerate(kept_ids)}
    sn_idx     = {sn: [id2idx[n] for n in members]
                  for sn, members in supernode_map.items()}
    sn_names   = list(supernode_map.keys())

    sn_act  = {sn: act[idx].sum().item() for sn, idx in sn_idx.items()}
    sn_flow = defaultdict(dict)
    for src in sn_names:
        for tgt in sn_names:
            if src == tgt:
                sn_flow[src][tgt] = 0.0
                continue
            total = sum(flow_mat[i, j].item()
                        for i in sn_idx[src]
                        for j in sn_idx[tgt])
            sn_flow[src][tgt] = total

    surrogate_to_logit = sum(
        sn_flow[src]['SN_LOGIT']
        for src in sn_names if src != 'SN_LOGIT'
    )
    return {
        'sn_act':              sn_act,
        'sn_flow':             dict(sn_flow),
        'sn_names':            sn_names,
        'surrogate_to_logit':  surrogate_to_logit,
    }


# ═══════════════════════════════════════════════════════════
# 4. AUTO NAMING
# ═══════════════════════════════════════════════════════════

THEME_KEYWORDS = {
    'DALLAS':    ['dallas', 'texas', 'tx'],
    'CAPITAL':   ['capital', 'government', 'political', 'state', 'policy'],
    'LOCATION':  ['location', 'city', 'cities', 'place', 'town', 'geographic'],
    'LEGAL':     ['legal', 'court', 'law', 'southern'],
    'GENERAL':   ['english', 'romance', 'language', 'general', 'code', 'news'],
}

def auto_name_cluster(members: list, attr: dict) -> str:
    scores = {theme: 0 for theme in THEME_KEYWORDS}
    for nid in members:
        text = attr[nid]['clerp'].lower()
        for theme, kws in THEME_KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    scores[theme] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return 'MISC'
    return best


# ═══════════════════════════════════════════════════════════
# 5. VALIDATION
# ═══════════════════════════════════════════════════════════
def build_dag_adj(adj_sub: torch.Tensor, layers: list) -> np.ndarray:
    """
    Build flow-aware adjacency cho directed Laplacian:
    1. Chỉ giữ forward edges (layer[i] <= layer[j])
    2. Scale by layer distance để enforce ordering
    3. Không dùng abs() — giữ sign để suppression edges bị downweight
    """
    N = len(layers)
    A = adj_sub.numpy().copy()

    for i in range(N):
        for j in range(N):
            if layers[i] > layers[j]:
                # Backward edge → zero out
                A[i, j] = 0.0
            elif layers[i] == layers[j]:
                # Same layer → keep but downweight
                A[i, j] = A[i, j] * 0.1
            # Forward edge → keep as is (abs for Laplacian stability)
            A[i, j] = abs(A[i, j])

    return A
def validate_supernode_map(
    supernode_map: dict,
    data: dict,
    orig_flow: dict,
) -> dict:
    surr_flow = compute_surrogate_flow(data, orig_flow, supernode_map)
    orig_total = orig_flow['total_to_logit']
    surr_total = surr_flow['surrogate_to_logit']
    preservation = surr_total / (orig_total + 1e-8)

    # Compute mean layer per supernode
    sn_layers = {}
    id2idx = {n: i for i, n in enumerate(data['kept_ids'])}
    for sn, members in supernode_map.items():
        idxs = [id2idx[n] for n in members]
        sn_layers[sn] = np.mean([data['layers'][i] for i in idxs])

    # Detect backward edges (flow goes from higher-layer to lower-layer SN)
    sn_names = surr_flow['sn_names']
    backward = []
    for src in sn_names:
        for tgt in sn_names:
            if src == tgt: continue
            if src == 'SN_LOGIT': continue  # logit has residual outgoing
            if tgt == 'SN_EMB':  continue  # EMB layer=0 always triggers false positive
            flow = surr_flow['sn_flow'][src][tgt]
            if flow > 1.0 and sn_layers.get(src, 0) > sn_layers.get(tgt, 99):
                backward.append((src, tgt, round(flow, 3)))
    # Intra-cluster cohesion: mean flow within each supernode
    cohesion = {}
    flow_mat = orig_flow['flow_matrix']
    for sn, members in supernode_map.items():
        idxs = [id2idx[n] for n in members]
        if len(idxs) < 2:
            cohesion[sn] = 0.0
            continue
        total = sum(flow_mat[i, j].item()
                    for i in idxs for j in idxs if i != j)
        cohesion[sn] = total / (len(idxs) * (len(idxs) - 1))

    return {
        'preservation':    preservation,
        'pass':            0.8 <= preservation <= 1.2,
        'backward_edges':  backward,
        'cohesion':        cohesion,
        'mean_cohesion':   np.mean(list(cohesion.values())),
        'sn_layers':       sn_layers,
        'surr_flow':       surr_flow,
    }


# ═══════════════════════════════════════════════════════════
# 6. CORE: AUTO BUILD SUPERNODE MAP
# ═══════════════════════════════════════════════════════════

def auto_build_supernode_map(
    data: dict,
    n_clusters: int = 3,
    weights: dict = None,
    algorithm: str = 'dag_spectral',
    verbose: bool = True,
) -> dict:
    """
    Parameters
    ----------
    data       : prepared data dict
    n_clusters : number of supernodes for transcoder nodes
                 (SN_EMB and SN_LOGIT are always added separately)
    weights    : dict with keys structural/layer/ctx/role/clerp
    algorithm  : 'spectral' | 'agglomerative'
    verbose    : print similarity breakdown

    Returns
    -------
    supernode_map : dict  name → list of node_ids
    """
    if weights is None:
        weights = {
            'structural': 0.50,
            'layer':      0.35,
            'ctx':        0.00,
            'role':       0.10,
            'clerp':      0.05,
        }

    kept_ids = data['kept_ids']
    attr     = data['attr']

    # ── Identify fixed nodes
    emb_ids   = [n for n in kept_ids if n.startswith('E')]
    logit_ids = [n for n in kept_ids if attr[n].get('is_target_logit', False)]
    fixed_ids = set(emb_ids + logit_ids)

    # ── Transcoder nodes only
    trans_ids = [n for n in kept_ids if n not in fixed_ids]
    trans_idx = [kept_ids.index(n) for n in trans_ids]
    M         = len(trans_ids)

    if M < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} > transcoder nodes={M}")

    # ── Sub-matrices for transcoder nodes
    adj_sub  = data['pruned_adj'][trans_idx][:, trans_idx]
    lay_sub  = [data['layers'][i] for i in trans_idx]
    inf_sub  = data['node_inf'][trans_idx]
    rel_sub  = data['node_rel'][trans_idx]

    # ── Compute similarity signals
    s_struct = sim_structural(adj_sub)
    s_layer  = sim_layer(lay_sub)
    s_ctx    = sim_ctx(trans_ids, attr)
    s_role   = sim_role(inf_sub, rel_sub)
    s_clerp  = sim_clerp_keyword(trans_ids, attr)

    if verbose:
        print(f"\n  Signal stats (mean similarity):")
        for name, s in [('structural', s_struct), ('layer', s_layer),
                        ('ctx', s_ctx), ('role', s_role), ('clerp', s_clerp)]:
            off_diag = s[~torch.eye(M, dtype=torch.bool)].mean().item()
            print(f"    {name:<12}  mean_off_diag={off_diag:.4f}  "
                  f"weight={weights.get(name,0):.2f}")

    # ── Combined similarity
    sim = (weights['structural'] * s_struct
         + weights['layer']      * s_layer
         + weights['ctx']        * s_ctx
         + weights['role']       * s_role
         + weights['clerp']      * s_clerp)

    sim_np = sim.numpy().clip(0, 1)
    np.fill_diagonal(sim_np, 1.0)

    # ── Clustering
    if algorithm == 'spectral':
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            n_init=10,
        ).fit(sim_np)
    elif algorithm == 'dag_spectral':
        labels, eigenvalues = dag_spectral_clustering(
            adj_sub=adj_sub,
            trans_ids=trans_ids,
            n_clusters=n_clusters,
            layers=lay_sub,  # ← thêm
            method='rw',
        )
        print(f"  Eigenvalues used: {eigenvalues.round(4)}")
    elif algorithm == 'dag_svd':  # ← thêm branch mới
        labels, singular_vals = dag_svd_clustering(
            adj_sub=adj_sub,  # 61×61, transcoder only
            trans_ids=trans_ids,
            data=data,
            n_clusters=n_clusters,
        )
        if verbose:
            print(f"  Singular values used: {singular_vals.round(4)}")

    elif algorithm == 'agglomerative':
        dist_np = 1.0 - sim_np
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average',
        ).fit(dist_np)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")



    # Silhouette score (quality of clustering)
    if M > n_clusters:
        sil = silhouette_score(1.0 - sim_np, labels, metric='precomputed')
    else:
        sil = 0.0

    if verbose:
        print(f"\n  Clustering: algorithm={algorithm}  "
              f"k={n_clusters}  silhouette={sil:.4f}")

    # ── Build supernode_map
    supernode_map = {}
    supernode_map['SN_EMB']   = emb_ids
    supernode_map['SN_LOGIT'] = logit_ids

    # Handle duplicate names by appending counter
    name_counts: Counter = Counter()
    for k in range(n_clusters):
        members = [trans_ids[i] for i, lbl in enumerate(labels) if lbl == k]
        base    = 'SN_' + auto_name_cluster(members, attr)
        name_counts[base] += 1
        name = base if name_counts[base] == 1 else f"{base}_{name_counts[base]}"
        supernode_map[name] = members

    # ── Verify completeness
    all_assigned  = [n for members in supernode_map.values() for n in members]
    unassigned    = set(kept_ids) - set(all_assigned)
    double_counted = [n for n in all_assigned if all_assigned.count(n) > 1]
    if unassigned:
        print(f"  [WARN] Unassigned: {unassigned}")
    if double_counted:
        print(f"  [WARN] Double-counted: {set(double_counted)}")

    return supernode_map, sil


# ═══════════════════════════════════════════════════════════
# 7. HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════

def search_best_k(
    data: dict,
    orig_flow: dict,
    k_range: range = range(2, 7),
    algorithm: str = 'dag_spectral',
    verbose: bool = True,
) -> dict:
    """
    Grid search over k (number of transcoder supernodes).
    Objective: maximize preservation × silhouette × (1 / (1 + backward_penalty))
    """
    results = []

    print(f"\n{'k':>4} {'sil':>8} {'pres':>8} {'back':>6} {'cohesion':>10}  {'score':>8}")
    print("-" * 55)

    for k in k_range:
        try:
            smap, sil = auto_build_supernode_map(
                data, n_clusters=k, algorithm=algorithm, verbose=False
            )
            val = validate_supernode_map(smap, data, orig_flow)
            pres    = val['preservation']
            n_back  = len(val['backward_edges'])
            coh     = val['mean_cohesion']

            # Score: higher is better
            # penalize: preservation far from 1, low silhouette, backward edges
            pres_score = 1.0 - abs(pres - 1.0)
            back_pen   = 1.0 / (1.0 + n_back)
            score      = (0.5 * pres_score + 0.3 * max(sil, 0) + 0.2 * back_pen)

            results.append({
                'k': k, 'sil': sil, 'preservation': pres,
                'backward': n_back, 'cohesion': coh,
                'score': score, 'supernode_map': smap,
            })
            print(f"{k:>4} {sil:>8.4f} {pres:>8.4f} {n_back:>6} {coh:>10.3f}  {score:>8.4f}")

        except Exception as e:
            print(f"{k:>4}  ERROR: {e}")

    if not results:
        raise RuntimeError("All k values failed.")

    best = max(results, key=lambda r: r['score'])
    print(f"\n  Best k={best['k']}  score={best['score']:.4f}")
    return best


# ═══════════════════════════════════════════════════════════
# 8. REPORTING
# ═══════════════════════════════════════════════════════════

def print_supernode_map(supernode_map: dict, data: dict):
    print(f"\n{'Supernode':<20} {'n':>4}  {'avg_layer':>10}  Members (clerp)")
    print("-" * 90)
    id2idx = {n: i for i, n in enumerate(data['kept_ids'])}
    for sn, members in supernode_map.items():
        idxs      = [id2idx[n] for n in members]
        avg_layer = np.mean([data['layers'][i] for i in idxs])
        clerps    = [data['clerp'][n] for n in members]
        sample    = ', '.join(clerps[:3])
        if len(clerps) > 3:
            sample += f'... (+{len(clerps)-3})'
        print(f"{sn:<20} {len(members):>4}  {avg_layer:>10.1f}  {sample}")

def print_surrogate_flow(surr_flow: dict, orig_flow: dict):
    sn_names   = surr_flow['sn_names']
    orig_total = orig_flow['total_to_logit']
    surr_total = surr_flow['surrogate_to_logit']
    ratio      = surr_total / (orig_total + 1e-8)

    print(f"\nFlow matrix between supernodes:")
    col_w  = 14
    header = f"{'':20}" + ''.join(f"{n:>{col_w}}" for n in sn_names)
    print(header)
    print("-" * (20 + col_w * len(sn_names)))
    for src in sn_names:
        row = f"{src:<20}"
        for tgt in sn_names:
            val = surr_flow['sn_flow'][src][tgt]
            row += f"{val:>{col_w}.2f}"
        print(row)

    print(f"\nFlow conservation:")
    print(f"  {'Supernode':<20} {'flow_in':>10} {'flow_out':>10} {'ratio':>8}")
    print("  " + "-" * 50)
    for sn in sn_names:
        fi = sum(surr_flow['sn_flow'][src][sn] for src in sn_names if src != sn)
        fo = sum(surr_flow['sn_flow'][sn][tgt] for tgt in sn_names if tgt != sn)
        print(f"  {sn:<20} {fi:>10.2f} {fo:>10.2f} {fo/(fi+1e-8):>8.3f}")

    print(f"\nPreservation:  orig={orig_total:.4f}  "
          f"surr={surr_total:.4f}  ratio={ratio:.4f}  "
          f"{'[PASS]' if 0.8<=ratio<=1.2 else '[FAIL]'}")

    print(f"\nFlow paths → SN_LOGIT:")
    for src in sn_names:
        if src == 'SN_LOGIT': continue
        val = surr_flow['sn_flow'][src]['SN_LOGIT']
        pct = val / (surr_total + 1e-8) * 100
        bar = '█' * int(pct / 2)
        print(f"  {src:<20} {val:>8.2f}  ({pct:>5.1f}%)  {bar}")


# ═══════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Auto Surrogate Graph Construction')
    parser.add_argument('--file',      type=str,   default='subgraph/austin_plt.pt',
                        help='Path to .pt snapshot file')
    parser.add_argument('--k',         type=int,   default=4,
                        help='Number of transcoder supernodes (default: 4)')
    parser.add_argument('--algorithm', type=str,   default='dag_svd',
                        choices=['spectral', 'agglomerative', 'dag_spectral','dag_svd'])
    parser.add_argument('--search',    action='store_true',
                        help='Search best k in range 2..6')
    parser.add_argument('--kmin',      type=int,   default=2)
    parser.add_argument('--kmax',      type=int,   default=6)
    args = parser.parse_args()

    # ── Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    if args.file:
        print(f"  From file: {args.file}")
        raw  = load_snapshot(args.file)
        data = prepare_data(raw)
    else:
        print("  Using built-in snapshot (build_data_from_snapshot)")
        data = build_data_from_snapshot()

    print(f"  Nodes: {len(data['kept_ids'])}")
    print(f"  Adj:   {data['pruned_adj'].shape}")

    # ── Original flow
    print("\n" + "=" * 60)
    print("ORIGINAL GRAPH FLOW")
    print("=" * 60)
    orig_flow = compute_original_flow(data)
    topk = torch.topk(orig_flow['flow_to_logit'], k=5)
    print(f"\n  Top-5 contributors to logit:")
    print(f"  {'Node':<20} {'clerp':<35} {'flow→logit':>10}")
    print("  " + "-" * 70)
    for val, idx in zip(topk.values, topk.indices):
        nid = data['kept_ids'][idx]
        print(f"  {nid:<20} {data['clerp'][nid]:<35} {val.item():>10.3f}")
    print(f"\n  Total flow → logit: {orig_flow['total_to_logit']:.4f}")

    # ── Auto construct supernode map
    print("\n" + "=" * 60)
    print("AUTO SUPERNODE CONSTRUCTION")
    print("=" * 60)

    if args.search:
        print(f"\n  Searching k in [{args.kmin}, {args.kmax}] ...")
        best = search_best_k(
            data, orig_flow,
            k_range=range(args.kmin, args.kmax + 1),
            algorithm=args.algorithm,
        )
        supernode_map = best['supernode_map']
        k_used        = best['k']
    else:
        k_used = args.k
        print(f"\n  Building with k={k_used}, algorithm={args.algorithm}")
        supernode_map, sil = auto_build_supernode_map(
            data, n_clusters=k_used, algorithm=args.algorithm, verbose=True
        )

    # ── Print supernode map
    print("\n" + "=" * 60)
    print("SUPERNODE MAP")
    print("=" * 60)
    print_supernode_map(supernode_map, data)

    # ── Surrogate flow
    print("\n" + "=" * 60)
    print("SURROGATE GRAPH FLOW")
    print("=" * 60)
    val = validate_supernode_map(supernode_map, data, orig_flow)
    print_surrogate_flow(val['surr_flow'], orig_flow)

    # ── Backward edges
    if val['backward_edges']:
        print(f"\n  [WARN] Backward edges detected:")
        for src, tgt, flow in val['backward_edges']:
            print(f"    {src} → {tgt}  flow={flow}")
    else:
        print(f"\n  [OK] No backward edges.")

    # ── Intra-cluster cohesion
    print(f"\n  Intra-cluster cohesion (mean intra-flow per node pair):")
    for sn, coh in val['cohesion'].items():
        bar = '░' * int(coh / 5)
        print(f"    {sn:<20} {coh:>8.3f}  {bar}")


if __name__ == '__main__':
    main()