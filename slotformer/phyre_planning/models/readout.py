import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.training import BaseModel
from nerv.models.transformer import build_pos_enc


class PHYREReadout(BaseModel):
    """A Transformer-based readout model for PHYRE.

    The goal of PHYRE is to predict whether two objects will contact as the
        scene evolves. The model takes in slots at multiple timesteps, use
        Transformer encoder to perform reasoning, and predicts a binary label.
    """

    def __init__(
        self,
        readout_dict=dict(
            num_slots=8,
            slot_size=128,
            t_pe='sin',
            d_model=128,
            num_layers=4,
            num_heads=8,
            ffn_dim=512,
            norm_first=True,
            sel_slots=[0, 3],  # reason over slots at selected timesteps
        ),
    ):
        super().__init__()

        self.readout_dict = readout_dict

        self._build_readout()

    def _build_readout(self):
        """Build readout head."""
        self.num_slots = self.readout_dict['num_slots']
        self.slot_size = self.readout_dict['slot_size']
        self.sel_slots = self.readout_dict['sel_slots']
        self.T = len(self.sel_slots)

        d_model = self.readout_dict['d_model']
        self.in_proj = nn.Linear(self.slot_size, d_model)
        self.CLS = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.readout_dict['num_heads'],
            dim_feedforward=self.readout_dict['ffn_dim'],
            norm_first=self.readout_dict['norm_first'],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=self.readout_dict['num_layers'],
        )
        self.enc_t_pe = build_pos_enc(self.readout_dict['t_pe'], self.T,
                                      d_model)

        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, data_dict):
        """Forward."""
        slots = data_dict['slots']
        slots = torch.stack([slots[:, i] for i in self.sel_slots], dim=1)

        slots = self.in_proj(slots)
        # apply temporal PE
        B, T, N, D = slots.shape
        slots = slots.flatten(1, 2)
        enc_pe = self.enc_t_pe.unsqueeze(2).repeat(B, 1, N, 1)
        slots = slots + enc_pe.flatten(1, 2)
        # concat CLS to get input tokens
        CLS = self.CLS.repeat(B, 1, 1)  # [B, 1, D]
        tokens = torch.cat([CLS, slots], dim=1)  # [B, 1 + (T*N), D]
        # relationship reasoning
        x = self.transformer_encoder(tokens)  # [B, 1 + (T*N), D]
        x = x[:, 0, :]  # [B, D], used for cls
        logits = self.cls_mlp(x)  # [B, 1]
        return {'logits': logits.squeeze(1)}  # [B]

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        pred = out_dict['logits'].flatten()
        gt = data_dict['label'].flatten().type_as(pred)
        vqa_loss = F.binary_cross_entropy_with_logits(pred, gt)
        loss_dict = {'vqa_loss': vqa_loss}
        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        ret_dict = self.calc_train_loss(data_dict, out_dict)
        # accuracy with different threshold
        pred = out_dict['logits'].flatten()
        gt = data_dict['label'].flatten().type_as(pred)
        acc_thresh = np.arange(0.1, 1, 0.2)
        pred_probs = torch.sigmoid(pred)
        for thresh in acc_thresh:
            eq_mask = (pred_probs > thresh).eq(gt)
            acc = eq_mask.float().mean()
            ret_dict[f'acc_{thresh:.2f}'] = acc
        return ret_dict

    @property
    def dtype(self):
        return self.CLS.dtype

    @property
    def device(self):
        return self.CLS.device
