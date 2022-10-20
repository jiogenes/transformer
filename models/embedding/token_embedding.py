import math
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out