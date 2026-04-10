import torch
raw = torch.load('subgraph/austin_plt.pt', map_location='cpu', weights_only=False)
print(raw.keys())