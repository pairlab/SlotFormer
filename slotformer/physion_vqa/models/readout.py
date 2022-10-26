import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.training import BaseModel


class PhysionReadout(BaseModel):
    """Linear readout model for Physion VQA task.

    Draw inspiration from RelationNetwork https://arxiv.org/pdf/1706.01427.pdf.
    We use a linear layer to compute the relation between all possible pairs of
        slots at one timestep. Then, we aggregate over all pairs by a symmetric
        function (sum/mean/max) to get the feature at this step. Finally, we
        take a max over all timesteps to get the final prediction.
    This is because, Physion VQA aims to predict whether two objects contact.
        The answer is true if any two objects contact at any timestep.
        So we take a max over all timesteps.
    """

    def __init__(
        self,
        readout_dict=dict(
            num_slots=6,
            slot_size=192,
            agg_func='max',
            feats_dim=192,
        ),
    ):
        super().__init__()

        self.readout_dict = readout_dict

        self._build_readout()

    def _build_readout(self):
        """Build readout head."""
        self.num_slots = self.readout_dict['num_slots']
        self.slot_size = self.readout_dict['slot_size']
        self.agg_func = self.readout_dict['agg_func']
        assert self.agg_func in ['sum', 'mean', 'max']
        feats_dim = self.readout_dict['feats_dim']

        # compute all possible combinations of slots
        combs = list(combinations(list(range(self.num_slots)), 2))
        comb_idx = torch.tensor(combs).long().flatten()  # [num_combs * 2]
        self.register_buffer('comb_idx', comb_idx)

        # linear layer to readout
        self.linear1 = nn.Linear(self.slot_size * 2, feats_dim)  # relation
        self.linear2 = nn.Linear(feats_dim, 1)  # logits

    def forward(self, data_dict):
        """Forward."""
        slots = data_dict['slots']  # [B, T, N, C]
        B, T = slots.shape[:2]
        slots = slots.flatten(0, 1)  # [B * T, N, C]
        comb_idx = self.comb_idx.to(slots.device)
        slots = slots[:, comb_idx]  # [B * T, num_combs * 2, C]
        slots = slots.unflatten(1, (-1, 2))  # [B * T, num_combs, 2, C]
        slots = slots.flatten(2, 3).unflatten(0, (B, -1))
        # [B, T, num_combs, 2 * C]
        relation = self.linear1(slots)  # [B, T, num_combs, feats_dim]
        # aggregate over object pairs --> [B, T, feats_dim]
        if self.agg_func == 'sum':
            relation = relation.sum(2)
        elif self.agg_func == 'mean':
            relation = relation.mean(2)
        else:
            relation = relation.max(2)[0]
        # predict logits --> [B, T, 1]
        logits = self.linear2(relation)  # [B, T, 1]
        # if two objects contact at one timestep, then the answer is yes
        # thus we take the max over time
        logits = logits.max(1)[0]  # [B, 1]
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
        return self.linear1.weight.dtype

    @property
    def device(self):
        return self.linear1.weight.device
