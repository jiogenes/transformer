from torch import nn

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed, dropout=0) -> None:
        super().__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.embedding(x)
        return out