import torch

from .slotformer import build_pos_enc, SlotRollouter, SlotFormer


class SingleStepSlotRollouter(SlotRollouter):
    """SlotRollouter with the iterative overlapping technique.

    Used in PHYRE, when the conditional input is just the 1st frame.
    Given I_0, we generate I_1; then we use [I_0, I_1] to generate I_2, etc.
    Until we get [I_0, ..., I_{cond_len}], then we'll generate the remaining
    slots autoregressively (same as SlotRollouter).
    """

    def __init__(
        self,
        num_slots,
        slot_size,
        history_len,  # burn-in steps, should be 1 in this model
        cond_len,  # this is the real `history_len` in `SlotRollouter` model
        t_pe='sin',
        slots_pe='',
        # Transformer-related configs
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
    ):
        super().__init__(
            num_slots=num_slots,
            slot_size=slot_size,
            history_len=history_len,
            t_pe=t_pe,
            slots_pe=slots_pe,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            norm_first=norm_first,
        )

        assert self.history_len == 1, \
            'SingleStepSlotRollouter performs rollout using only initial frame'
        self.cond_len = cond_len
        self.num_cond_tokens = self.cond_len * self.num_slots
        self.enc_t_pe = build_pos_enc(t_pe, cond_len, d_model)  # `cond_len`

    def forward(self, x, pred_len):
        """Forward function.

        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: int

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len

        B = x.shape[0]
        x = x.flatten(1, 2)  # [B, history_len * N, slot_size]
        in_x = x

        # temporal_pe repeat for each slot, shouldn't be None
        # [B, cond_len * N, C]
        enc_pe = self.enc_t_pe.unsqueeze(2).\
            repeat(B, 1, self.num_slots, 1).flatten(1, 2)
        # slots_pe repeat for each timestep
        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1).\
                repeat(B, self.cond_len, 1, 1).flatten(1, 2)
            enc_pe = slots_pe + enc_pe

        pred_out = []
        for _ in range(pred_len):
            # we take last `num_cond_tokens` to predict slots
            # project to latent space
            x = self.in_proj(in_x[:, -self.num_cond_tokens:])
            # encoder positional encoding
            x = x + enc_pe[:, -x.shape[1]:]
            # spatio-temporal interaction via Transformer
            x = self.transformer_encoder(x)
            # take the last N output tokens to predict slots
            pred_slots = self.out_proj(x[:, -self.num_slots:])
            pred_out.append(pred_slots)
            # feed the predicted slots autoregressively
            in_x = torch.cat([in_x, pred_out[-1]], dim=1)

        return torch.stack(pred_out, dim=1)


class SingleStepSlotFormer(SlotFormer):
    """Transformer-based rollouter on slot embeddings."""

    def _build_loss(self):
        super()._build_loss()
        # a hack, on PHYRE we'll equip SlotFormer with a task success cls model
        self.use_cls_loss = False
        self.success_cls = None

    def _build_rollouter(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Rollouter
        self.history_len = self.rollout_dict['history_len']  # 1
        self.rollouter = SingleStepSlotRollouter(**self.rollout_dict)

    def classify(self, slots, vid_len=None):
        """Task success classifier."""
        # only used in PHYRE eval
        assert not self.training
        # slots: [B, T, N, C]
        logits = self.success_cls({
            'slots': slots,
            'vid_len': vid_len,
        })['logits']
        return logits  # [B]

    def forward(self, data_dict):
        """Forward pass."""
        out_dict = super().forward(data_dict)
        if not (self.use_cls_loss and self.success_cls is not None):
            return out_dict
        pred_slots = out_dict['pred_slots']
        past_slots = out_dict['gt_slots']
        slots = torch.cat([past_slots, pred_slots], dim=1)
        vid_len = data_dict.get('vid_len', None)
        out_dict['logits'] = self.classify(slots, vid_len)
        return out_dict
