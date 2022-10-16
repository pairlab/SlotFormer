"""Transition function used in SAVi and STEVE."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass


class TransformerPredictor(Predictor):
    """Transformer encoder."""

    def __init__(
        self,
        d_model=128,
        num_layers=1,
        num_heads=4,
        ffn_dim=256,
        norm_first=True,
    ):
        super().__init__()

        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer, num_layers=num_layers)

    def forward(self, x):
        out = self.transformer_encoder(x)
        return out


class ResidualMLPPredictor(Predictor):
    """LN + residual MLP."""

    def __init__(self, channels, norm_first=True):
        super().__init__()

        assert len(channels) >= 2
        # since there is LN at the beginning of slot-attn
        # so only use a pre-ln here
        self.ln = nn.LayerNorm(channels[0])
        modules = []
        for i in range(len(channels) - 2):
            modules += [nn.Linear(channels[i], channels[i + 1]), nn.ReLU()]
        modules.append(nn.Linear(channels[-2], channels[-1]))
        self.mlp = nn.Sequential(*modules)

        self.norm_first = norm_first

    def forward(self, x):
        if not self.norm_first:
            res = x
        x = self.ln(x)
        if self.norm_first:
            res = x
        out = self.mlp(x)
        out = out + res
        return out


class RNNPredictorWrapper(Predictor):
    """Predictor wrapped in a RNN for sequential modeling."""

    def __init__(
        self,
        base_predictor,
        input_size=128,
        hidden_size=256,
        num_layers=1,
        rnn_cell='LSTM',
        sg_every=None,
    ):
        super().__init__()

        assert rnn_cell in ['LSTM', 'GRU', 'RNN']
        self.base_predictor = base_predictor
        self.rnn = eval(f'nn.{rnn_cell.upper()}(input_size={input_size}, '
                        f'hidden_size={hidden_size}, num_layers={num_layers})')
        self.step = 0
        self.hidden_state = None
        self.out_projector = nn.Linear(hidden_size, input_size)
        self.sg_every = sg_every  # detach all inputs every certain steps
        # in ICCV'21 PARTS (https://openaccess.thecvf.com/content/ICCV2021/papers/Zoran_PARTS_Unsupervised_Segmentation_With_Slots_Attention_and_Independence_Maximization_ICCV_2021_paper.pdf)
        # they detach RNN states every 4 steps to avoid overfitting
        # but we don't observe much difference in our experiments

    def forward(self, x):
        if self.sg_every is not None:
            if self.step % self.sg_every == 0 and self.step > 0:
                x = x.detach()
                # LSTM hiddens state is (h, c) tuple
                if not isinstance(self.hidden_state, torch.Tensor):
                    self.hidden_state = tuple(
                        [h.detach() for h in self.hidden_state])
                else:
                    self.hidden_state = self.hidden_state.detach()
        # `x` should have shape of [B, ..., C]
        out = self.base_predictor(x)
        out_shape = out.shape
        self.rnn.flatten_parameters()
        out, self.hidden_state = self.rnn(
            out.view(1, -1, out_shape[-1]), self.hidden_state)
        out = self.out_projector(out[0]).view(out_shape)
        self.step += 1
        return out

    def burnin(self, x):
        """Warm up the RNN by first few steps inputs."""
        self.reset()
        # `x` should have shape of [B, T, ..., C]
        B, T = x.shape[:2]
        out = self.base_predictor(x.flatten(0, 1)).unflatten(0, (B, T))
        out = out.transpose(1, 0).reshape(T, -1, x.shape[-1])
        _, self.hidden_state = self.rnn(out, self.hidden_state)
        self.step = T

    def reset(self):
        """Clear the RNN hidden state."""
        self.step = 0
        self.hidden_state = None
