"""
Circuit Flow Analysis
Original graph vs Surrogate graph
Data: austin_plt.pt (gemma-2-2b circuit for "The capital of the state containing Dallas is")
"""

import torch
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def parse_layer(nid):
    if nid.startswith('E'):  return 0
    return int(nid.split('_')[0])  # '16', '17', ..., '27'

def parse_activation(attr_node):
    """
    Transcoder nodes: activation = raw float
    Embedding nodes:  activation = None → dùng influence * scale
    Logit node:       activation = None, is_target_logit=True → dùng token_prob * 100
    """
    a = attr_node['activation']
    if a is not None:
        return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100  # 0.4504 * 100 = 45.04
    # embedding: estimate từ influence
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100

def parse_influence(attr_node):
    """
    influence = adj[i, logit_idx] = direct edge weight to logit
    Logit node itself: None (no outgoing edge to itself)
    """
    inf = attr_node.get('influence')
    return float(inf) if inf is not None else 0.0

def load_snapshot(path: str) -> dict:
    return torch.load(path, map_location="cpu",weights_only=True)

# If running locally:
# data = load_snapshot('subgraph/austin_plt.pt')

# For reproducibility, we reconstruct from the known snapshot values
# (replace with data = load_snapshot(...) when file is available)

def build_data_from_snapshot():
    """
    Reconstruct data dict from the known snapshot.
    In practice: data = torch.load('subgraph/austin_plt.pt', map_location='cpu')
    """
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
        '27_22605_10'
    ]

    # activation values (None → estimated)
    act_values = {
        '16_25_9': 28.16,    '16_12678_9': 32.06,  '16_4298_10': 19.93,
        '16_13497_10': 10.26,'17_7178_10': 27.50,  '18_1026_10': 15.99,
        '18_1437_10': 9.26,  '18_3852_10': 8.45,   '18_5495_10': 12.45,
        '18_6101_10': 20.99, '18_8959_10': 39.96,  '18_16041_10': 9.47,
        '19_7477_9': 55.78,  '19_37_10': 9.70,     '19_1445_10': 35.45,
        '19_2439_10': 17.40, '19_2695_10': 17.39,  '19_7477_10': 26.97,
        '20_15589_9': 45.66, '20_114_10': 17.25,   '20_5916_10': 53.43,
        '20_6026_10': 17.02, '20_7507_10': 8.21,   '20_15276_10': 9.98,
        '20_15589_10': 49.56,'21_5943_10': 56.37,  '21_6316_10': 10.86,
        '21_6795_10': 16.33, '21_14975_10': 12.31, '22_31_10': 24.70,
        '22_3064_10': 58.50, '22_3551_10': 48.36,  '22_4999_10': 37.47,
        '22_11718_10': 34.40,'23_2288_10': 14.86,  '23_6617_10': 9.10,
        '23_11444_10': 31.55,'23_12237_10': 54.44, '23_12918_10': 18.48,
        '23_13193_10': 26.53,'23_13541_10': 13.38, '23_13841_10': 23.95,
        '23_15366_10': 23.05,'24_709_10': 10.41,   '24_6044_10': 55.07,
        '24_6394_10': 25.96, '24_13277_10': 116.57,'24_15013_10': 29.57,
        '24_15627_10': 26.43,'24_15694_10': 15.49, '24_16258_10': 22.16,
        '25_553_10': 51.21,  '25_583_10': 25.03,   '25_762_10': 11.22,
        '25_2687_10': 10.40, '25_4259_10': 29.21,  '25_4679_10': 16.61,
        '25_4717_10': 31.33, '25_4886_10': 27.92,  '25_13300_10': 22.51,
        '25_16302_10': 39.76,
        # Embeddings: estimated from influence * 100
        'E_6037_4': 4.15,    'E_2329_7': 22.77,   'E_26865_9': 17.14,
        # Logit: token_prob * 100
        '27_22605_10': 45.04,
    }

    influence_values = {
        '16_25_9': 0.3085,   '16_12678_9': 0.4794,  '16_4298_10': 0.4371,
        '16_13497_10': 0.5397,'17_7178_10': 0.4327,  '18_1026_10': 0.4726,
        '18_1437_10': 0.3878, '18_3852_10': 0.5543,  '18_5495_10': 0.3778,
        '18_6101_10': 0.3535, '18_8959_10': 0.3443,  '18_16041_10': 0.5711,
        '19_7477_9': 0.4462,  '19_37_10': 0.4213,    '19_1445_10': 0.3703,
        '19_2439_10': 0.4290, '19_2695_10': 0.3482,  '19_7477_10': 0.3802,
        '20_15589_9': 0.3150, '20_114_10': 0.5014,   '20_5916_10': 0.4005,
        '20_6026_10': 0.5161, '20_7507_10': 0.4665,  '20_15276_10': 0.4912,
        '20_15589_10': 0.2618,'21_5943_10': 0.2895,  '21_6316_10': 0.5390,
        '21_6795_10': 0.3928, '21_14975_10': 0.4778, '22_31_10': 0.4495,
        '22_3064_10': 0.3551, '22_3551_10': 0.3996,  '22_4999_10': 0.3403,
        '22_11718_10': 0.3581,'23_2288_10': 0.4864,  '23_6617_10': 0.4076,
        '23_11444_10': 0.3977,'23_12237_10': 0.2938, '23_12918_10': 0.4414,
        '23_13193_10': 0.3919,'23_13541_10': 0.4482, '23_13841_10': 0.4110,
        '23_15366_10': 0.4101,'24_709_10': 0.5278,   '24_6044_10': 0.2694,
        '24_6394_10': 0.3181, '24_13277_10': 0.3207, '24_15013_10': 0.3967,
        '24_15627_10': 0.5335,'24_15694_10': 0.3857, '24_16258_10': 0.4341,
        '25_553_10': 0.3909,  '25_583_10': 0.4400,   '25_762_10': 0.5573,
        '25_2687_10': 0.5372, '25_4259_10': 0.3813,  '25_4679_10': 0.4768,
        '25_4717_10': 0.5019, '25_4886_10': 0.4710,  '25_13300_10': 0.3118,
        '25_16302_10': 0.4198,
        'E_6037_4': 0.0415,   'E_2329_7': 0.2277,   'E_26865_9': 0.1714,
        '27_22605_10': None,
    }

    clerp = {
        '16_25_9': 'Texas legal documents',
        '16_12678_9': 'cities',
        '16_4298_10': 'capital',
        '16_13497_10': 'Numbers and parameters',
        '17_7178_10': 'government buildings',
        '18_1026_10': 'country names',
        '18_1437_10': 'Legal documents from Texas',
        '18_3852_10': 'Locations',
        '18_5495_10': 'locations',
        '18_6101_10': 'capital cities',
        '18_8959_10': 'government/state',
        '18_16041_10': 'capital',
        '19_7477_9': 'Dallas',
        '19_37_10': 'Places',
        '19_1445_10': 'Downtowns of cities',
        '19_2439_10': 'Politics and government',
        '19_2695_10': 'cities',
        '19_7477_10': 'Dallas',
        '20_15589_9': 'Texas',
        '20_114_10': 'Oklahoma locations',
        '20_5916_10': 'capital',
        '20_6026_10': 'political titles',
        '20_7507_10': 'countries',
        '20_15276_10': 'Dallas sports',
        '20_15589_10': 'Texas',
        '21_5943_10': 'cities',
        '21_6316_10': 'special',
        '21_6795_10': 'geographic place names near Texas',
        '21_14975_10': 'state/states',
        '22_31_10': 'government and policy',
        '22_3064_10': 'Texas/Dallas',
        '22_3551_10': 'Place names and legal cases',
        '22_4999_10': 'Locations',
        '22_11718_10': 'Texas locations',
        '23_2288_10': 'Texas',
        '23_6617_10': 'Locations',
        '23_11444_10': 'cities and places',
        '23_12237_10': 'Cities and states names',
        '23_12918_10': 'Texas',
        '23_13193_10': 'Legal and Southern place names',
        '23_13541_10': 'News articles',
        '23_13841_10': 'towns and cities',
        '23_15366_10': 'Code snippets',
        '24_709_10': 'patent identifiers',
        '24_6044_10': 'in',
        '24_6394_10': 'locations',
        '24_13277_10': 'Romance languages',
        '24_15013_10': 'in',
        '24_15627_10': 'Locations',
        '24_15694_10': 'US states',
        '24_16258_10': 'Detects place names',
        '25_553_10': 'general English text',
        '25_583_10': 'city names',
        '25_762_10': 'international locations',
        '25_2687_10': 'locations',
        '25_4259_10': 'place names',
        '25_4679_10': 'locations',
        '25_4717_10': 'unusual/fantastical narratives',
        '25_4886_10': 'last names',
        '25_13300_10': 'Texas',
        '25_16302_10': 'Legal/court cases',
        'E_6037_4': 'Emb: " capital"',
        'E_2329_7': 'Emb: " state"',
        'E_26865_9': 'Emb: " Dallas"',
        '27_22605_10': 'Output " Austin" (p=0.450)',
    }

    # Build pruned_adj from influence values (adj[i, logit_idx] = influence[i])
    # For other edges, we use a synthetic sparse matrix that preserves the
    # layer ordering and known direct-to-logit weights.
    # In practice: load from torch.load(...)
    N = len(kept_ids)
    adj = torch.zeros(N, N)
    logit_idx = N - 1  # index 64

    # Set direct-to-logit edges from influence values
    for i, nid in enumerate(kept_ids):
        inf = influence_values.get(nid)
        if inf is not None:
            adj[i, logit_idx] = inf

    # Set inter-layer edges based on layer ordering and activation
    # (synthetic but layer-consistent for flow verification)
    def get_layer(nid):
        if nid.startswith('E'): return 0
        if nid.startswith('27'): return 27
        return int(nid.split('_')[0])

    layers = [get_layer(n) for n in kept_ids]

    torch.manual_seed(42)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if j == logit_idx: continue  # already set
            li, lj = layers[i], layers[j]
            # Only allow forward edges (layer i < layer j)
            if li < lj:
                # Synthetic weight proportional to activation ratio
                w = (act_values[kept_ids[i]] / 200.0) * torch.rand(1).item()
                if w > 0.05:  # sparse threshold
                    adj[i, j] = round(w, 4)

    return {
        'kept_ids': kept_ids,
        'pruned_adj': adj,
        'act_values': act_values,
        'influence_values': influence_values,
        'clerp': clerp,
        'layers': layers,
    }


# ─────────────────────────────────────────────
# 2. ORIGINAL GRAPH FLOW
# ─────────────────────────────────────────────

def compute_original_flow(data: dict) -> dict:
    kept_ids   = data['kept_ids']
    adj        = data['pruned_adj']          # (N, N)
    act_values = data['act_values']
    N          = len(kept_ids)

    # Activation vector
    act = torch.tensor([act_values[n] for n in kept_ids], dtype=torch.float32)

    # flow_matrix[i,j] = act[i] * adj[i,j]
    flow_matrix = act.unsqueeze(1) * adj     # (N, N)

    # Per-node aggregates
    flow_out = flow_matrix.sum(dim=1)        # row sum
    flow_in  = flow_matrix.sum(dim=0)        # col sum

    # Direct flow to logit (act * influence)
    logit_idx = N - 1
    flow_to_logit = flow_matrix[:, logit_idx]

    # Top-k contributors to logit
    topk = torch.topk(flow_to_logit, k=10)

    print("=" * 60)
    print("ORIGINAL GRAPH FLOW")
    print("=" * 60)

    print(f"\n{'Node':<20} {'clerp':<35} {'act':>8} {'inf':>6} {'flow→logit':>10}")
    print("-" * 85)
    for rank, (val, idx) in enumerate(zip(topk.values, topk.indices)):
        nid = kept_ids[idx]
        print(f"{nid:<20} {data['clerp'][nid]:<35} "
              f"{act_values[nid]:>8.2f} "
              f"{data['influence_values'].get(nid, 0):>6.3f} "
              f"{val.item():>10.3f}")

    total_to_logit = flow_to_logit.sum().item()
    print(f"\nTotal flow → logit:  {total_to_logit:.4f}")

    print(f"\n{'Node':<20} {'layer':>6} {'flow_in':>10} {'flow_out':>10} {'ratio':>8}")
    print("-" * 60)
    for i, nid in enumerate(kept_ids):
        fi = flow_in[i].item()
        fo = flow_out[i].item()
        ratio = fo / (fi + 1e-8)
        if fi > 1.0 or fo > 1.0:  # skip near-zero nodes
            print(f"{nid:<20} {data['layers'][i]:>6} "
                  f"{fi:>10.3f} {fo:>10.3f} {ratio:>8.3f}")

    return {
        'act': act,
        'flow_matrix': flow_matrix,
        'flow_in': flow_in,
        'flow_out': flow_out,
        'flow_to_logit': flow_to_logit,
        'total_to_logit': total_to_logit,
        'logit_idx': logit_idx,
    }


# ─────────────────────────────────────────────
# 3. DEFINE SURROGATE GRAPH
# ─────────────────────────────────────────────

def define_surrogate(kept_ids: list) -> dict:
    """
    5 supernodes + 2 fixed (embedding, logit)
    Returns supernode_map: name → list of member node_ids
    """
    supernode_map = {
        'SN_EMB': [
            'E_6037_4', 'E_2329_7', 'E_26865_9',
        ],
        'SN_DALLAS': [
            # ctx=9 nodes + Texas/Dallas clerp (layers 16-22)
            '16_25_9',       # Texas legal docs
            '16_12678_9',    # cities
            '19_7477_9',     # Dallas (ctx=9)
            '20_15589_9',    # Texas (ctx=9)
            '19_7477_10',    # Dallas (ctx=10)
            '20_15589_10',   # Texas
            '21_6795_10',    # geographic place names near Texas
            '22_3064_10',    # Texas/Dallas
            '22_11718_10',   # Texas locations
            '23_2288_10',    # Texas
            '23_12918_10',   # Texas
            '25_13300_10',   # Texas
        ],
        'SN_CAPITAL': [
            # capital/government clerp (layers 16-23)
            '16_4298_10',    # capital
            '16_13497_10',   # Numbers/parameters
            '17_7178_10',    # government buildings
            '18_1026_10',    # country names
            '18_6101_10',    # capital cities
            '18_8959_10',    # government/state
            '18_16041_10',   # capital
            '19_2439_10',    # Politics/government
            '20_5916_10',    # capital
            '20_6026_10',    # political titles
            '21_14975_10',   # state/states
            '22_31_10',      # government and policy
            '23_12237_10',   # Cities and states names
        ],
        'SN_LOCATION': [
            # late-layer location/place nodes (output prep)
            '18_1437_10', '18_3852_10', '18_5495_10',
            '19_37_10',   '19_1445_10', '19_2695_10',
            '20_114_10',  '20_7507_10', '20_15276_10',
            '21_5943_10', '21_6316_10',
            '22_3551_10', '22_4999_10',
            '23_6617_10', '23_11444_10', '23_13193_10',
            '23_13541_10','23_13841_10', '23_15366_10',
            '24_709_10',  '24_6044_10',  '24_6394_10',
            '24_13277_10','24_15013_10', '24_15627_10',
            '24_15694_10','24_16258_10',
            '25_553_10',  '25_583_10',   '25_762_10',
            '25_2687_10', '25_4259_10',  '25_4679_10',
            '25_4717_10', '25_4886_10',  '25_16302_10',
        ],
        'SN_LOGIT': [
            '27_22605_10',
        ],
    }

    # Verify: every node in kept_ids is assigned exactly once
    all_assigned = []
    for members in supernode_map.values():
        all_assigned.extend(members)

    assigned_set   = set(all_assigned)
    kept_set       = set(kept_ids)
    unassigned     = kept_set - assigned_set
    double_counted = [n for n in all_assigned if all_assigned.count(n) > 1]

    if unassigned:
        print(f"[WARN] Unassigned nodes: {unassigned}")
    if double_counted:
        print(f"[WARN] Double-counted nodes: {set(double_counted)}")
    if not unassigned and not double_counted:
        print(f"[OK] All {len(kept_ids)} nodes assigned to exactly one supernode.")

    return supernode_map


# ─────────────────────────────────────────────
# 4. SURROGATE GRAPH FLOW
# ─────────────────────────────────────────────

def compute_surrogate_flow(data: dict, orig_flow: dict, supernode_map: dict) -> dict:
    kept_ids    = data['kept_ids']
    flow_matrix = orig_flow['flow_matrix']   # (N, N)
    act         = orig_flow['act']           # (N,)
    N           = len(kept_ids)

    # Index lookup
    id2idx = {nid: i for i, nid in enumerate(kept_ids)}

    # Supernode indices
    sn_idx = {
        sn: [id2idx[n] for n in members]
        for sn, members in supernode_map.items()
    }

    sn_names = list(supernode_map.keys())

    # ── Supernode activation = sum of member activations
    sn_act = {
        sn: act[idx].sum().item()
        for sn, idx in sn_idx.items()
    }

    # ── Supernode flow between pairs
    # sn_flow[src][tgt] = Σ_{i∈src, j∈tgt} flow_matrix[i,j]
    sn_flow = defaultdict(dict)
    for src in sn_names:
        for tgt in sn_names:
            if src == tgt:
                sn_flow[src][tgt] = 0.0
                continue
            total = 0.0
            for i in sn_idx[src]:
                for j in sn_idx[tgt]:
                    total += flow_matrix[i, j].item()
            sn_flow[src][tgt] = total

    # ── Surrogate flow matrix (5×5)
    M = len(sn_names)
    sn_flow_matrix = np.zeros((M, M))
    for r, src in enumerate(sn_names):
        for c, tgt in enumerate(sn_names):
            sn_flow_matrix[r, c] = sn_flow[src][tgt]

    # ── Print surrogate flow table
    print("\n" + "=" * 60)
    print("SURROGATE GRAPH FLOW")
    print("=" * 60)

    print(f"\nSupernode activations:")
    for sn, a in sn_act.items():
        members = supernode_map[sn]
        print(f"  {sn:<15} n={len(members):>3}  act_sum={a:>8.2f}")

    print(f"\nFlow matrix between supernodes (rows=src, cols=tgt):")
    col_w = 14
    header = f"{'':15}" + "".join(f"{n:>{col_w}}" for n in sn_names)
    print(header)
    print("-" * (15 + col_w * M))
    for r, src in enumerate(sn_names):
        row = f"{src:<15}"
        for c, tgt in enumerate(sn_names):
            val = sn_flow_matrix[r, c]
            row += f"{val:>{col_w}.3f}"
        print(row)

    # ── Flow conservation per supernode
    print(f"\nFlow conservation check:")
    print(f"  {'Supernode':<15} {'flow_in':>10} {'flow_out':>10} {'ratio':>8}")
    print("  " + "-" * 46)
    for sn in sn_names:
        fi = sum(sn_flow[src][sn] for src in sn_names if src != sn)
        fo = sum(sn_flow[sn][tgt] for tgt in sn_names if tgt != sn)
        ratio = fo / (fi + 1e-8)
        print(f"  {sn:<15} {fi:>10.3f} {fo:>10.3f} {ratio:>8.3f}")

    # ── Total flow to logit
    surrogate_to_logit = sum(
        sn_flow[src]['SN_LOGIT']
        for src in sn_names if src != 'SN_LOGIT'
    )

    return {
        'sn_act': sn_act,
        'sn_flow': dict(sn_flow),
        'sn_flow_matrix': sn_flow_matrix,
        'sn_names': sn_names,
        'surrogate_to_logit': surrogate_to_logit,
    }


# ─────────────────────────────────────────────
# 5. PRESERVATION CHECK
# ─────────────────────────────────────────────

def check_preservation(orig_flow: dict, surr_flow: dict):
    orig_total = orig_flow['total_to_logit']
    surr_total = surr_flow['surrogate_to_logit']
    ratio      = surr_total / (orig_total + 1e-8)

    print("\n" + "=" * 60)
    print("FLOW PRESERVATION CHECK")
    print("=" * 60)
    print(f"\n  Original  total flow → logit : {orig_total:.4f}")
    print(f"  Surrogate total flow → logit : {surr_total:.4f}")
    print(f"  Preservation ratio           : {ratio:.4f}")

    if 0.8 <= ratio <= 1.2:
        print(f"  [PASS] Surrogate preserves flow within 20% tolerance.")
    elif 0.6 <= ratio < 0.8:
        print(f"  [WARN] Surrogate loses ~{(1-ratio)*100:.1f}% of flow.")
    else:
        print(f"  [FAIL] Surrogate deviates significantly from original.")

    # Per-path breakdown
    print(f"\n  Flow paths into SN_LOGIT:")
    sn_flow = surr_flow['sn_flow']
    sn_names = surr_flow['sn_names']
    for src in sn_names:
        if src == 'SN_LOGIT': continue
        val = sn_flow[src]['SN_LOGIT']
        pct = val / (surr_total + 1e-8) * 100
        bar = '█' * int(pct / 2)
        print(f"    {src:<15} {val:>8.3f}  ({pct:>5.1f}%)  {bar}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("Loading data...")
    # Option A: load from file

    raw = load_snapshot('subgraph/austin_plt.pt',)

    # Inject influence vào cột logit
    adj = raw['pruned_adj'].clone()
    logit_idx = len(raw['kept_ids']) - 1
    for i, nid in enumerate(raw['kept_ids']):
        inf = raw['attr'][nid].get('influence')
        if inf is not None:
            adj[i, logit_idx] = float(inf)

    data = {
        'kept_ids': raw['kept_ids'],
        'pruned_adj': adj,
        'act_values': {n: parse_activation(raw['attr'][n])
                       for n in raw['kept_ids']},
        'influence_values': {n: parse_influence(raw['attr'][n])
                             for n in raw['kept_ids']},
        'clerp': {n: raw['attr'][n]['clerp']
                  for n in raw['kept_ids']},
        'layers': [parse_layer(n) for n in raw['kept_ids']],
    }
    # ── Option B: reconstruct from snapshot values
    #data = build_data_from_snapshot()
    print(f"  Nodes: {len(data['kept_ids'])}")
    print(f"  Adj shape: {data['pruned_adj'].shape}")

    # Step 2: Original flow
    orig_flow = compute_original_flow(data)

    # Step 3: Define surrogate
    print("\n" + "=" * 60)
    print("DEFINING SURROGATE GRAPH")
    print("=" * 60)
    supernode_map = define_surrogate(data['kept_ids'])

    # Step 4: Surrogate flow
    surr_flow = compute_surrogate_flow(data, orig_flow, supernode_map)

    # Step 5: Preservation
    check_preservation(orig_flow, surr_flow)