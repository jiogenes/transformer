from copy import deepcopy
from torch import nn

from models.layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, norm, dropout=0):
        super().__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer(deepcopy(norm), dropout) for _ in range(2)]

    def forward(self, src, src_mask):
        x = self.residuals[0](src, lambda src: self.self_attention(query=src, key=src, value=src, mask=src_mask))
        x = self.residuals[1](src, self.position_ff)
        return x