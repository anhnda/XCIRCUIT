def check_order_constraint(supernode_map, data, orig_flow, threshold=1.0):
    """
    Kiểm tra ORDER CONSTRAINT cho supernode_map:
    ∀ u∈C_i, v∈C_j: nếu flow(u→v) > threshold
                     thì avg_layer(C_i) < avg_layer(C_j)
    """
    kept_ids = data['kept_ids']
    id2idx   = {n: i for i, n in enumerate(kept_ids)}
    flow_mat = orig_flow['flow_matrix']

    # Compute avg_layer per supernode
    sn_layers = {}
    sn_layer_range = {}
    for sn, members in supernode_map.items():
        idxs = [id2idx[n] for n in members]
        ls   = [data['layers'][i] for i in idxs]
        sn_layers[sn]      = np.mean(ls)
        sn_layer_range[sn] = (min(ls), max(ls))

    sn_names = list(supernode_map.keys())

    print(f"\n{'Supernode':<20} {'avg_layer':>10} {'layer_range':>15}  {'n':>4}")
    print("-" * 55)
    for sn in sn_names:
        lo, hi = sn_layer_range[sn]
        print(f"  {sn:<18} {sn_layers[sn]:>10.1f} "
              f"  [{lo:>2}, {hi:>2}]        {len(supernode_map[sn]):>4}")

    print(f"\nChecking ORDER CONSTRAINT (threshold={threshold}):")
    violations = []
    ok_count   = 0

    for src in sn_names:
        for tgt in sn_names:
            if src == tgt: continue
            if src == 'SN_EMB' or tgt == 'SN_EMB': continue
            if src == 'SN_LOGIT' or tgt == 'SN_LOGIT': continue

            # Tính total flow src → tgt
            src_idx = [id2idx[n] for n in supernode_map[src]]
            tgt_idx = [id2idx[n] for n in supernode_map[tgt]]
            flow    = sum(flow_mat[i, j].item()
                         for i in src_idx for j in tgt_idx)

            if abs(flow) <= threshold:
                continue

            # Kiểm tra: nếu flow > 0 thì avg_layer(src) < avg_layer(tgt)?
            if flow > threshold:
                if sn_layers[src] > sn_layers[tgt]:
                    violations.append({
                        'src': src, 'tgt': tgt,
                        'flow': flow,
                        'src_layer': sn_layers[src],
                        'tgt_layer': sn_layers[tgt],
                        'src_range': sn_layer_range[src],
                        'tgt_range': sn_layer_range[tgt],
                    })
                else:
                    ok_count += 1

    if not violations:
        print(f"  [PASS] All {ok_count} forward flows respect layer ordering.")
    else:
        print(f"  [FAIL] {len(violations)} violations found:")
        print(f"\n  {'src':<20} {'tgt':<20} {'flow':>10} "
              f"{'src_avg':>8} {'tgt_avg':>8}  layer_ranges")
        print("  " + "-" * 85)
        for v in violations:
            print(f"  {v['src']:<20} {v['tgt']:<20} {v['flow']:>10.1f} "
                  f"{v['src_layer']:>8.1f} {v['tgt_layer']:>8.1f}  "
                  f"{v['src_range']} → {v['tgt_range']}")

    # Check layer range overlap between supernodes
    print(f"\nChecking layer range overlap:")
    overlap_found = False
    for i, sn_i in enumerate(sn_names):
        for j, sn_j in enumerate(sn_names):
            if j <= i: continue
            if 'EMB' in sn_i or 'LOGIT' in sn_i: continue
            if 'EMB' in sn_j or 'LOGIT' in sn_j: continue
            lo_i, hi_i = sn_layer_range[sn_i]
            lo_j, hi_j = sn_layer_range[sn_j]
            overlap = max(0, min(hi_i, hi_j) - max(lo_i, lo_j) + 1)
            if overlap > 0:
                overlap_found = True
                print(f"  [OVERLAP] {sn_i} {sn_layer_range[sn_i]} ∩ "
                      f"{sn_j} {sn_layer_range[sn_j]} = {overlap} layers")

    if not overlap_found:
        print(f"  [OK] No layer range overlap between supernodes.")

    return violations


# Predefined map từ đầu conversation
predefined_map = {
    'SN_EMB': ['E_6037_4', 'E_2329_7', 'E_26865_9'],
    'SN_DALLAS': [
        '16_25_9', '16_12678_9', '19_7477_9', '20_15589_9',
        '19_7477_10', '20_15589_10', '21_6795_10', '22_3064_10',
        '22_11718_10', '23_2288_10', '23_12918_10', '25_13300_10',
    ],
    'SN_CAPITAL': [
        '16_4298_10', '16_13497_10', '17_7178_10', '18_1026_10',
        '18_6101_10', '18_8959_10', '18_16041_10', '19_2439_10',
        '20_5916_10', '20_6026_10', '21_14975_10', '22_31_10',
        '23_12237_10',
    ],
    'SN_LOCATION': [
        '18_1437_10', '18_3852_10', '18_5495_10', '19_37_10',
        '19_1445_10', '19_2695_10', '20_114_10', '20_7507_10',
        '20_15276_10', '21_5943_10', '21_6316_10', '22_3551_10',
        '22_4999_10', '23_6617_10', '23_11444_10', '23_13193_10',
        '23_13541_10', '23_13841_10', '23_15366_10', '24_709_10',
        '24_6044_10', '24_6394_10', '24_13277_10', '24_15013_10',
        '24_15627_10', '24_15694_10', '24_16258_10', '25_553_10',
        '25_583_10', '25_762_10', '25_2687_10', '25_4259_10',
        '25_4679_10', '25_4717_10', '25_4886_10', '25_16302_10',
    ],
    'SN_LOGIT': ['27_22605_10'],
}
from auto_surrogate import *
data = build_data_from_snapshot()
orig_flow = compute_original_flow(data)
violations = check_order_constraint(predefined_map, data, orig_flow)