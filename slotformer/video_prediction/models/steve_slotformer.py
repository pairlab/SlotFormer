import torch
import torch.nn.functional as F

from nerv.training import BaseModel

from .slotformer import SlotFormer
from slotformer.base_slots.models import gumbel_softmax, make_one_hot, STEVE


class STEVESlotFormer(SlotFormer):
    """Transformer-based rollouter on slot embeddings."""

    def __init__(
            self,
            resolution,
            clip_len,
            slot_dict=dict(
                num_slots=6,
                slot_size=192,
            ),
            dvae_dict=dict(
                down_factor=4,
                vocab_size=4096,
                dvae_ckp_path='',
            ),
            dec_dict=dict(
                dec_num_layers=4,
                dec_num_heads=4,
                dec_d_model=192,
                dec_ckp_path='',
            ),
            rollout_dict=dict(
                num_slots=6,
                slot_size=192,
                history_len=6,
                t_pe='sin',
                slots_pe='',
                d_model=192,
                num_layers=4,
                num_heads=8,
                ffn_dim=192 * 4,
                norm_first=True,
            ),
            loss_dict=dict(
                rollout_len=6,
                use_img_recon_loss=False,
            ),
            eps=1e-6,
    ):
        self.dvae_dict = dvae_dict

        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            dec_dict=dec_dict,
            rollout_dict=rollout_dict,
            loss_dict=loss_dict,
            eps=eps,
        )

    def _build_dvae(self):
        # Build the same dVAE model as in STEVE
        STEVE._build_dvae(self)

    def _build_decoder(self):
        # Build dVAE first because Decoder relies on it
        self._build_dvae()
        # Build the same Transformer decoder as in STEVE
        STEVE._build_decoder(self)
        # name it as `decoder` for consistency
        import copy
        self.decoder = copy.deepcopy(self.trans_decoder)  # TODO: better way?
        del self.trans_decoder
        # load pretrained weight
        ckp_path = self.dec_dict['dec_ckp_path']
        assert ckp_path, 'Please provide pretrained Transformer decoder weight'
        w = torch.load(ckp_path, map_location='cpu')['state_dict']
        w = {k[14:]: v for k, v in w.items() if k.startswith('trans_decoder.')}
        self.decoder.load_state_dict(w)
        # freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

    def decode(self, slots):
        """Decode from slots to reconstructed images.
        This function is super slot when the resolution is 128, because the
            the patch token sequence will be 32x32=1024.
        """
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        _, logits = self.decoder.generate(
            slots, steps=self.num_patches, sample=False)
        # [B, patch_size**2, vocab_size] --> [B, vocab_size, h, w]
        logits = logits.transpose(2, 1).\
            unflatten(-1, (self.h, self.w)).contiguous().cuda()
        z_logits = F.log_softmax(logits, dim=1)
        z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
        soft_recon = self.dvae.detokenize(z)  # [B, C, H, W]
        # SLATE directly use ont-hot token as reconstruction input
        z_hard = make_one_hot(logits, dim=1)
        hard_recon = self.dvae.detokenize(z_hard)
        return soft_recon, hard_recon

    def rollout(self, past_slots, pred_len, decode=False, with_gt=True):
        """Perform future rollout to `target_len` video."""
        pred_slots = self.rollouter(past_slots[:, -self.history_len:],
                                    pred_len)
        return pred_slots

    def forward(self, data_dict):
        """Forward pass."""
        slots = data_dict['slots']  # [B, T', N, C]
        assert self.rollout_len + self.history_len == slots.shape[1], \
            f'wrong SlotFormer training length {slots.shape[1]}'
        past_slots = slots[:, :self.history_len]
        gt_slots = slots[:, self.history_len:]
        pred_slots = self.rollout(past_slots, self.rollout_len)
        out_dict = {
            'gt_slots': gt_slots,  # both slots [B, pred_len, N, C]
            'pred_slots': pred_slots,
        }

        # `img_recon_loss` is actually the token reconstruction loss in STEVE
        # note that this is very memory-consuming
        if self.use_img_recon_loss:
            img = data_dict['img']  # [B, T', C, H, W]
            gt_img = img[:, self.history_len:]  # [B, T, C, H, W]
            # tokenize the images
            if 'token_id' in data_dict:
                gt_token_id = data_dict['token_id']
            else:
                with torch.no_grad():
                    gt_token_id = self.dvae.tokenize(
                        gt_img, one_hot=False).flatten(2, 3).detach()
            h, w = self.h, self.w
            target_token_id = gt_token_id.flatten(0, 1).long()  # [B*T, H*W]
            # TransformerDecoder token prediction loss
            in_slots = pred_slots.flatten(0, 1)  # [B*T, N, C]
            in_token_id = target_token_id[:, :-1]
            pred_token_id = self.decoder(in_slots, in_token_id)[:, -(h * w):]
            # [B*T, h*w, vocab_size]
            out_dict.update({
                'pred_token_id': pred_token_id,
                'target_token_id': target_token_id,
            })

        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        gt_slots = out_dict['gt_slots']
        pred_slots = out_dict['pred_slots']
        slot_recon_loss = F.mse_loss(pred_slots, gt_slots)
        loss_dict = {'slot_recon_loss': slot_recon_loss}
        if self.use_img_recon_loss:
            pred_token_id = out_dict['pred_token_id'].flatten(0, 1)
            target_token_id = out_dict['target_token_id'].flatten(0, 1)
            token_recon_loss = F.cross_entropy(pred_token_id, target_token_id)
            loss_dict['img_recon_loss'] = token_recon_loss
        return loss_dict

    def train(self, mode=True):
        BaseModel.train(self, mode)
        # keep dVAE and Transformer decoder in eval mode
        self.dvae.eval()
        self.decoder.eval()
        return self
