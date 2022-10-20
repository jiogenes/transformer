import torch
from torch import nn

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2, dropout=0) -> None:
        super().__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU()
        self.fc2 = fc2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x