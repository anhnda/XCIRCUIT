import torch
from visualize_circuit_sp_rep import parse_layer, prepare_graph_data
raw = torch.load('subgraph/austin_clt.pt', map_location='cpu', weights_only=False)
print(raw)
#data = prepare_graph_data(raw)
# logit_idx = data['logit_idx']
# print("Logit outgoing row after .T:", data['adj'][logit_idx, :])
#
# for nid in raw['kept_ids']:
#     print(nid, parse_layer(nid))