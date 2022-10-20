from torch import nn

class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, dropout=0) -> None:
        super().__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, sub_layer):
        out = self.norm(x)
        out = sub_layer(out)
        out = self.dropout(out)
        out += x
        return out