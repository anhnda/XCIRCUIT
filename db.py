import torch
from visualize_circuit_sp_rep import parse_layer
raw = torch.load('subgraph/austin_clt.pt', map_location='cpu', weights_only=False)
print(raw)
for nid in raw['kept_ids']:
    print(nid, parse_layer(nid))