import torch
import torch.nn as nn


def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding."""
    if pos_enc == 'learnable':
        # ViT, BEiT etc. all use zero-init learnable pos enc
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    elif pos_enc == 'sin':
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding


def build_transformer(
    input_len,
    pos_enc,
    d_model,
    num_heads,
    ffn_dim,
    num_layers,
    norm_first=True,
):
    """Build the Transformer Encoder.

    Args:
        norm_first (bool): whether apply pre-LN
    """
    transformer_enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        norm_first=norm_first,
        batch_first=True,
    )
    transformer_encoder = TransformerEncoderWithPosEnc(
        input_len=input_len,
        pos_enc=pos_enc,
        d_model=d_model,
        encoder_layer=transformer_enc_layer,
        num_layers=num_layers,
    )
    return transformer_encoder


class TransformerEncoderWithPosEnc(nn.TransformerEncoder):
    """TransformerEncoder with positional encoding at input."""

    def __init__(
        self,
        input_len,
        pos_enc,
        d_model,
        encoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__(encoder_layer, num_layers, norm)

        # build positional encoding
        self.pos_embedding = build_pos_enc(pos_enc, input_len, d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """Apply PE and then the normal forward."""
        src = src + self.pos_embedding
        return super().forward(src, mask, src_key_padding_mask)
