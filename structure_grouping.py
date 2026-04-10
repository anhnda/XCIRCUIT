"""
structure_grouping.py — minimal version for testing flow_analysis.py
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


def parse_layer(nid: str) -> int:
    if nid.startswith('E'):   return 0
    if nid.startswith('27'):  return 27
    return int(nid.split('_')[0])

def parse_activation(attr_node: dict) -> float:
    a = attr_node.get('activation')
    if a is not None: return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100

def parse_influence(attr_node: dict) -> float:
    inf = attr_node.get('influence')
    return float(inf) if inf is not None else 0.0

def load_snapshot(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)

def prepare_graph_data(raw: dict) -> dict:
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float().T
    attr      = raw['attr']
    logit_idx = len(kept_ids) - 1
    layers     = [parse_layer(n) for n in kept_ids]
    act_values = {n: parse_activation(attr[n]) for n in kept_ids}
    inf_values = {n: parse_influence(attr[n])  for n in kept_ids}
    clerp      = {n: attr[n].get('clerp', '')  for n in kept_ids}
    inf_to_logit = {nid: float(attr[nid].get('influence') or 0.0) for nid in kept_ids}
    if 'node_inf' in raw:
        node_inf = raw['node_inf'].float()
    else:
        ni = torch.tensor([inf_values[n] for n in kept_ids], dtype=torch.float32)
        node_inf = ni / (ni.max() + 1e-8)
    return dict(kept_ids=kept_ids, adj=adj, act_values=act_values,
                inf_values=inf_values, inf_to_logit=inf_to_logit,
                clerp=clerp, layers=layers, node_inf=node_inf, logit_idx=logit_idx)

def _cosine_norm(M: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.diag(M).clamp(min=1e-8))
    return M / diag.unsqueeze(1) / diag.unsqueeze(0)

def compute_similarity(data, alpha=0.5, beta=0.5):
    kept_ids, adj = data['kept_ids'], data['adj']
    act_t = torch.tensor([data['act_values'][n] for n in kept_ids], dtype=torch.float32)
    inf_t = torch.tensor([data['inf_values'][n] for n in kept_ids], dtype=torch.float32)
    act_norm = act_t / (act_t.max() + 1e-8)
    inf_norm = inf_t / (inf_t.max() + 1e-8)
    W = torch.diag(alpha * act_norm + beta * inf_norm)
    S_out_cos = _cosine_norm(adj @ W @ adj.T)
    S_in_cos  = _cosine_norm(adj.T @ W @ adj)
    return (0.5 * S_out_cos + 0.5 * S_in_cos).clamp(0.0, 1.0)

def _is_fixed(nid: str) -> bool:
    return nid.startswith('E') or nid.startswith('27')

def cluster_middle_nodes(data, S, threshold=0.50, linkage_method='average'):
    kept_ids = data['kept_ids']
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    S_mid = S[middle_idx][:, middle_idx].numpy()
    D_mid = (1.0 - S_mid).clip(0.0, 1.0)
    Z = linkage(squareform(D_mid, checks=False), method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion='distance')
    raw_clusters = defaultdict(list)
    for nid, lbl in zip(middle_ids, labels):
        raw_clusters[int(lbl)].append(nid)
    return dict(raw_clusters=dict(raw_clusters), Z=Z, middle_ids=middle_ids, D_mid=D_mid)

def cluster_with_target_k(data, S, target_k=7, max_layer_span=4, max_sn=None):
    from sklearn.cluster import SpectralClustering
    kept_ids = data['kept_ids']
    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    M = len(middle_ids)
    if target_k >= M:
        raise ValueError(f'target_k={target_k} >= middle nodes={M}')
    S_mid = ((S[middle_idx][:, middle_idx].numpy() + S[middle_idx][:, middle_idx].numpy().T) / 2).clip(0, 1)
    upper = S_mid[np.triu_indices(M, k=1)]
    global_mean = float(upper.mean())
    print(f'  Similarity  mean={global_mean:.4f}  median={np.median(upper):.4f}'
          f'  p75={np.percentile(upper,75):.4f}  max={upper.max():.4f}')
    outlier_mask = S_mid.max(axis=1) < global_mean
    core_local = [i for i in range(M) if not outlier_mask[i]]
    outlier_ids = [middle_ids[i] for i in range(M) if outlier_mask[i]]
    core_ids = [middle_ids[i] for i in core_local]
    if outlier_ids:
        print(f'  Outliers ({len(outlier_ids)}): {outlier_ids}')
    S_core = S_mid[np.ix_(core_local, core_local)]
    effective_k = min(target_k, len(core_ids) - 1)
    if effective_k < 2:
        raw_clusters = {0: core_ids}
    else:
        labels = SpectralClustering(n_clusters=effective_k, affinity='precomputed',
                                     assign_labels='kmeans', random_state=42, n_init=20).fit_predict(S_core)
        raw_clusters = defaultdict(list)
        for nid, lbl in zip(core_ids, labels):
            raw_clusters[int(lbl)].append(nid)
    layers = {nid: parse_layer(nid) for nid in middle_ids}
    if outlier_ids:
        cluster_mean_layer = {lbl: np.mean([layers[n] for n in members]) for lbl, members in raw_clusters.items()}
        for nid in outlier_ids:
            nearest = min(cluster_mean_layer, key=lambda l: abs(cluster_mean_layer[l] - layers[nid]))
            raw_clusters[nearest].append(nid)
    print(f'  Spectral clusters (before DAG enforcement): {len(raw_clusters)}')
    return enforce_dag(dict(raw_clusters), data, max_layer_span, max_sn)

def merge_to_budget(final, layers, max_sn):
    while len(final) > max_sn:
        best_i, best_j, best_gap = 0, 1, float('inf')
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                ci = np.mean([layers[n] for n in final[i]])
                cj = np.mean([layers[n] for n in final[j]])
                if abs(ci - cj) < best_gap:
                    best_i, best_j, best_gap = i, j, abs(ci - cj)
        final[best_i] = final[best_i] + final[best_j]
        final.pop(best_j)
        final.sort(key=lambda m: min(layers[n] for n in m))
    return final

def enforce_dag(raw_clusters, data, max_layer_span=4, max_sn=None):
    layers = {nid: parse_layer(nid) for members in raw_clusters.values() for nid in members}
    queue = list(raw_clusters.values())
    final = []
    while queue:
        members = queue.pop()
        lvals = sorted(set(layers[n] for n in members))
        if lvals[-1] - lvals[0] <= max_layer_span or len(members) == 1:
            final.append(members)
        else:
            mid = (lvals[0] + lvals[-1]) / 2
            lower = [n for n in members if layers[n] <= mid]
            upper = [n for n in members if layers[n] > mid]
            if not lower or not upper:
                half = len(members) // 2
                lower, upper = members[:half], members[half:]
            queue.extend([lower, upper])
    final.sort(key=lambda m: min(layers[n] for n in m))
    changed = True
    while changed:
        changed = False
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                lo_i, hi_i = min(layers[n] for n in final[i]), max(layers[n] for n in final[i])
                lo_j, hi_j = min(layers[n] for n in final[j]), max(layers[n] for n in final[j])
                if (lo_i < lo_j < hi_i) or (lo_j < lo_i < hi_j):
                    victim = i if (hi_i-lo_i) >= (hi_j-lo_j) else j
                    other = j if victim == i else i
                    split = min(layers[n] for n in final[other])
                    lo_part = [n for n in final[victim] if layers[n] < split]
                    hi_part = [n for n in final[victim] if layers[n] >= split]
                    if lo_part and hi_part:
                        final[victim] = lo_part
                        final.append(hi_part)
                        final.sort(key=lambda m: min(layers[n] for n in m))
                        changed = True
                        break
            if changed: break
    if max_sn is not None and len(final) > max_sn:
        final = merge_to_budget(final, layers, max_sn)
    final_supernodes = {}
    for idx, members in enumerate(final):
        lo, hi = min(layers[n] for n in members), max(layers[n] for n in members)
        name = f'SN_{idx:02d}_L{lo}' if lo == hi else f'SN_{idx:02d}_L{lo}_{hi}'
        final_supernodes[name] = members
    for nid in [n for n in data['kept_ids'] if n.startswith('E')]:
        final_supernodes[f'SN_EMB_{nid}'] = [nid]
    for nid in [n for n in data['kept_ids'] if n.startswith('27')]:
        final_supernodes[f'SN_LOGIT_{nid}'] = [nid]
    return final_supernodes

def check_dag_safety(final_supernodes):
    def layer_range(members):
        lvals = [parse_layer(n) for n in members]
        return min(lvals), max(lvals)
    ranges = {sn: layer_range(m) for sn, m in final_supernodes.items()}
    sn_list = list(ranges.keys())
    warnings = []
    for i, sn_a in enumerate(sn_list):
        for sn_b in sn_list[i+1:]:
            lo_a, hi_a = ranges[sn_a]
            lo_b, hi_b = ranges[sn_b]
            if lo_a < lo_b < hi_a < hi_b: warnings.append((sn_a, sn_b))
            elif lo_b < lo_a < hi_b < hi_a: warnings.append((sn_b, sn_a))
    return warnings

def evaluate_grouping(final_supernodes, data, S):
    kept_ids, act_values, clerp = data['kept_ids'], data['act_values'], data['clerp']
    id2idx = {nid: i for i, nid in enumerate(kept_ids)}
    stats = {}
    for sn, members in final_supernodes.items():
        idx = [id2idx[n] for n in members if n in id2idx]
        lvals = [parse_layer(n) for n in members]
        pairs = [(i, j) for ii, i in enumerate(idx) for j in idx[ii+1:]]
        if pairs:
            sims = [S[i, j].item() for i, j in pairs]
            intra_mean, intra_min = float(np.mean(sims)), float(np.min(sims))
        else:
            intra_mean = intra_min = 1.0
        stats[sn] = dict(n=len(members), layer_lo=min(lvals), layer_hi=max(lvals),
                          layer_span=max(lvals)-min(lvals),
                          act_max=max(act_values.get(n, 0) for n in members),
                          intra_sim_mean=intra_mean, intra_sim_min=intra_min,
                          members=members, clerps=[clerp.get(n, '') for n in members])
    return stats

def build_supernode_graph(final_supernodes, data):
    kept_ids, adj, act_values = data['kept_ids'], data['adj'], data['act_values']
    inf_to_logit, layers = data['inf_to_logit'], data['layers']
    id2idx = {nid: i for i, nid in enumerate(kept_ids)}
    sn_names = list(final_supernodes.keys())
    K = len(sn_names)
    sn_act = {sn: max((act_values.get(n, 0.0) for n in m), default=0.0) for sn, m in final_supernodes.items()}
    act_max_global = max(sn_act.values()) or 1.0
    sn_act_norm = {sn: v / act_max_global for sn, v in sn_act.items()}
    sn_inf = {sn: sum(inf_to_logit.get(n, 0.0) for n in m) for sn, m in final_supernodes.items()}
    sn_adj_mat = np.zeros((K, K), dtype=np.float64)
    for i, sn_a in enumerate(sn_names):
        idx_a = [id2idx[n] for n in final_supernodes[sn_a] if n in id2idx]
        if not idx_a: continue
        for j, sn_b in enumerate(sn_names):
            if i == j: continue
            idx_b = [id2idx[n] for n in final_supernodes[sn_b] if n in id2idx]
            if not idx_b: continue
            src_t, tgt_t = torch.tensor(idx_a, dtype=torch.long), torch.tensor(idx_b, dtype=torch.long)
            block = adj[src_t][:, tgt_t]
            src_layers = torch.tensor([layers[s] for s in idx_a])
            tgt_layers = torch.tensor([layers[t] for t in idx_b])
            fwd_mask = (src_layers.unsqueeze(1) <= tgt_layers.unsqueeze(0)).float()
            sn_adj_mat[i, j] = (block * fwd_mask).sum().item()
    total_inf_orig = sum(inf_to_logit.get(n, 0.0) for n in kept_ids)
    total_inf_sn = sum(sn_inf.values())
    inf_conservation = total_inf_sn / (total_inf_orig + 1e-12)
    node2sn = {nid: sn for sn, m in final_supernodes.items() for nid in m}
    total_fwd_orig = sum(adj[i, j].item() for i in range(len(kept_ids)) for j in range(len(kept_ids))
                         if i != j and layers[i] <= layers[j] and adj[i, j].item() != 0.0
                         and node2sn.get(kept_ids[i]) != node2sn.get(kept_ids[j]))
    total_fwd_sn = float(sn_adj_mat.sum())
    edge_conservation = total_fwd_sn / (total_fwd_orig + 1e-12)
    edges = []
    for i, src in enumerate(sn_names):
        for j, tgt in enumerate(sn_names):
            if i != j and sn_adj_mat[i, j] > 0:
                edges.append(dict(src=src, tgt=tgt, weight=float(sn_adj_mat[i, j])))
    edges.sort(key=lambda e: e['weight'], reverse=True)
    dominant_paths = edges[:5]
    excit_in = np.array([sum(v for v in sn_adj_mat[:, j] if v > 0) for j in range(K)])
    total_inf = sum(sn_inf.values()) or 1.0
    pos_in = excit_in[excit_in > 0]
    med_excit_in = float(np.median(pos_in)) if len(pos_in) > 0 else 0.0
    bottleneck_sns = [sn_names[i] for i in range(K)
                      if excit_in[i] > med_excit_in and sn_inf.get(sn_names[i], 0) / total_inf > 0.05
                      and sn_names[i] not in ('SN_LOGIT', 'SN_EMB')]
    sn_inf_arr = np.array([sn_inf[sn] for sn in sn_names])
    sn_act_arr = np.array([sn_act[sn] for sn in sn_names])
    sn_act_norm_arr = np.array([sn_act_norm[sn] for sn in sn_names])
    return dict(sn_names=sn_names, sn_act=sn_act_arr, sn_act_norm=sn_act_norm_arr,
                sn_inf=sn_inf_arr, sn_adj=sn_adj_mat, F_sn=sn_adj_mat,
                sn_reach=sn_inf_arr, inf_conservation=inf_conservation,
                edge_conservation=edge_conservation, total_inf_orig=total_inf_orig,
                total_inf_sn=total_inf_sn, total_fwd_orig=total_fwd_orig,
                total_fwd_sn=total_fwd_sn, dominant_paths=dominant_paths,
                bottleneck_sns=bottleneck_sns, preservation=inf_conservation,
                orig_reach_total=total_inf_orig, surr_reach_total=total_inf_sn)

def print_report(final_supernodes, stats, sng, dag_warnings):
    print('  [report]')

def build_synthetic_snapshot():
    kept_ids = [
        '16_25_9', '16_12678_9', '16_4298_10', '16_13497_10', '17_7178_10',
        '18_1026_10', '18_1437_10', '18_3852_10', '18_5495_10', '18_6101_10',
        '18_8959_10', '18_16041_10', '19_7477_9', '19_37_10', '19_1445_10',
        '19_2439_10', '19_2695_10', '19_7477_10', '20_15589_9', '20_114_10',
        '20_5916_10', '20_6026_10', '20_7507_10', '20_15276_10', '20_15589_10',
        '21_5943_10', '21_6316_10', '21_6795_10', '21_14975_10',
        '22_31_10', '22_3064_10', '22_3551_10', '22_4999_10', '22_11718_10',
        '23_2288_10', '23_6617_10', '23_11444_10', '23_12237_10', '23_12918_10',
        '23_13193_10', '23_13541_10', '23_13841_10', '23_15366_10',
        '24_709_10', '24_6044_10', '24_6394_10', '24_13277_10', '24_15013_10',
        '24_15627_10', '24_15694_10', '24_16258_10',
        '25_553_10', '25_583_10', '25_762_10', '25_2687_10', '25_4259_10',
        '25_4679_10', '25_4717_10', '25_4886_10', '25_13300_10', '25_16302_10',
        'E_6037_4', 'E_2329_7', 'E_26865_9', '27_22605_10',
    ]
    act_v = {'16_25_9':28.16,'16_12678_9':32.06,'16_4298_10':19.93,'16_13497_10':10.26,
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
        '27_22605_10':45.04}
    inf_v = {'16_25_9':0.3085,'16_12678_9':0.4794,'16_4298_10':0.4371,'16_13497_10':0.5397,
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
        '27_22605_10':None}
    clerp = {'16_25_9':'Texas legal documents','16_12678_9':'cities',
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
        '27_22605_10':'Output " Austin" (p=0.450)'}
    N = len(kept_ids)
    adj = torch.zeros(N, N)
    logit_idx = N - 1
    layers = [parse_layer(n) for n in kept_ids]
    torch.manual_seed(42)
    for i in range(N):
        for j in range(N):
            if i == j or j == logit_idx: continue
            if layers[i] < layers[j]:
                w = (act_v[kept_ids[i]] / 200.0) * torch.rand(1).item()
                if w > 0.05: adj[i, j] = round(w, 4)
    attr = {}
    for nid in kept_ids:
        is_emb, is_log = nid.startswith('E'), nid.startswith('27')
        attr[nid] = dict(activation=act_v[nid] if not (is_emb or is_log) else None,
                          influence=inf_v.get(nid), clerp=clerp[nid],
                          is_target_logit=is_log, token_prob=0.4504 if is_log else None,
                          ctx_idx=int(nid.split('_')[2]) if '_' in nid and not is_emb else 0)
    return dict(kept_ids=kept_ids, pruned_adj=adj, attr=attr)