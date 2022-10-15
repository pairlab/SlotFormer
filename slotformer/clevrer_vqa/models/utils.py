from nerv.models.transformer import build_transformer_encoder


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
    return build_transformer_encoder(
        input_len=input_len,
        pos_enc=pos_enc,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        norm_first=norm_first,
        norm_last=False,
    )
