import copy
from torch import nn

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer, norm):
        super().__init__()
        self.layers = [copy.deepcopy(encoder_block) for _ in range(n_layer)]
        self.norm = norm

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out