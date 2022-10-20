import copy
from torch import nn

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer, norm) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)
        return out