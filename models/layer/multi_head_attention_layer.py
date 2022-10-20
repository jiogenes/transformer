import copy
import math
import torch
from torch import nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, qkv_fc, out_fc, dropout=0):
        super().__init__()
        # d_model == h * d_k
        self.d_model = d_model
        self.heads = heads
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, query, key, value, mask):
        # query, key, value : (batch_size, heads, seq_len, d_k)
        # mask : (batch_size, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # (batch_size, heads, seq_len, seq_len)
        attention_score /= math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = torch.nn.functional.softmax(attention_score, dim=-1) # (batch_size, heads, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        x = torch.matmul(attention_prob, value) # (batch_size, heads, seq_len, d_k)
        return x

    def forward(self, query, key, value, mask=None):
        # query, key, value : (batch_size, seq_len, d_embed)
        # mask : (batch_size, seq_len, seq_len)
        # return value : (batch_size, h, seq_len, d_k)
        batch_size = query.size(0)

        def transform(x, fc): # (batch_size, seq_len, d_embed)
            x = fc(x) # (batch_size, seq_len, d_model)
            x = x.view(batch_size, -1, self.heads, self.d_model//self.heads) # (batch_size, seq_len, heads, d_k)
            x = x.transpose(1, 2) # (batch_size, heads, seq_len, d_k)
            return x
        
        query = transform(query, self.q_fc) # (batch_size, heads, seq_len, d_k)
        key = transform(key, self.k_fc)     # (batch_size, heads, seq_len, d_k)
        value = transform(value, self.v_fc) # (batch_size, heads, seq_len, d_k)

        x = self.calculate_attention(query, key, value, mask) # (batch_size, heads, seq_len, d_k)
        x = x.transpose(1, 2) # (batch_size, seq_len, heads, d_k)
        x = x.contiguous().view(batch_size, -1, self.d_model) # (batch_size, heads, d_model)
        x = self.out_fc(x) # (batch_size, heads, d_model)
        return x
        


