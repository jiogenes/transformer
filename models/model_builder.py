import torch
from torch import nn
from copy import deepcopy

from models.encoder_decoder.decoder import Decoder
from models.encoder_decoder.encoder import Encoder
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.transformer_embedding import TransformerEmbedding
from models.encoder_decoder.decoder_block import DecoderBlock
from models.encoder_decoder.encoder_block import EncoderBlock
from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.transformer import Transformer

def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device('cpu'),
                max_len=256,
                d_embed=512,
                n_layer=6,
                d_model=512,
                heads=8,
                d_ff=2048,
                dropout=0.1,
                norm_eps=1e-5):

    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=tgt_vocab_size)

    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len, device=device)
    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=deepcopy(pos_embed), dropout=dropout)
    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed, pos_embed=deepcopy(pos_embed), dropout=dropout)

    attention = MultiHeadAttentionLayer(d_model=d_model, heads=heads, qkv_fc=nn.Linear(d_model, d_model), out_fc=nn.Linear(d_model, d_embed), dropout=dropout)
    position_ff = PositionWiseFeedForwardLayer(fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed), dropout=dropout)

    norm = nn.LayerNorm(d_embed, eps=norm_eps)

    encoder_block = EncoderBlock(self_attention=deepcopy(attention), position_ff=deepcopy(position_ff), norm=deepcopy(norm), dropout=dropout)
    decoder_block = DecoderBlock(self_attention=deepcopy(attention), cross_attention=deepcopy(attention), position_ff=deepcopy(position_ff), norm=deepcopy(norm), dropout=dropout)

    encoder = Encoder(encoder_block=encoder_block, n_layer=n_layer, norm=deepcopy(norm))
    decoder = Decoder(decoder_block=decoder_block, n_layer=n_layer, norm=deepcopy(norm))

    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(src_embed=src_embed, tgt_embed=tgt_embed, encoder=encoder, decoder=decoder, generator=generator).to(device)

    model.device = device

    return model