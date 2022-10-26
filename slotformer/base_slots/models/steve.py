import torch
from torch import nn
from torch.nn import functional as F

from nerv.training import BaseModel
from .utils import torch_cat
from .savi import SlotAttention, StoSAVi
from .dVAE import dVAE
from .steve_transformer import STEVETransformerDecoder
from .steve_utils import gumbel_softmax


class SlotAttentionWMask(SlotAttention):
    """Slot attention module that iteratively performs cross-attention.

    We return the last attention map from SA as the segmentation mask.
    """

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            slots (torch.Tensor): [B, num_slots, C] slot inits.
            masks (torch.Tensor): [B, num_slots, N] segmentation mask.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        # Multiple rounds of attention.
        for attn_iter in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # attn_map normalized along slot-dim is treated as seg_mask
            if attn_iter == self.num_iterations - 1:
                seg_mask = attn.detach().clone().permute(0, 2, 1)

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        return slots, seg_mask


class STEVE(StoSAVi):
    """SA model with TransformerDecoder predicting patch tokens."""

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
        ),
        dvae_dict=dict(
            down_factor=4,
            vocab_size=4096,
            dvae_ckp_path='',
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_type='slate',
            dec_num_layers=4,
            dec_num_heads=4,
            dec_d_model=128,
        ),
        pred_dict=dict(
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_img_recon_loss=False,  # dVAE decoded img recon loss
        ),
        eps=1e-6,
    ):
        BaseModel.__init__(self)

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps

        self.slot_dict = slot_dict
        self.dvae_dict = dvae_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict
        self.loss_dict = loss_dict

        self._build_slot_attention()
        self._build_dvae()
        self._build_encoder()
        self._build_decoder()
        self._build_predictor()
        self._build_loss()

        # a hack for only extracting slots
        self.testing = False

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        self.slot_attention = SlotAttentionWMask(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_dvae(self):
        # Build dVAE module
        self.vocab_size = self.dvae_dict['vocab_size']
        self.down_factor = self.dvae_dict['down_factor']
        self.dvae = dVAE(vocab_size=self.vocab_size, img_channels=3)
        ckp_path = self.dvae_dict['dvae_ckp_path']
        assert ckp_path, 'Please provide pretrained dVAE weight'
        ckp = torch.load(ckp_path, map_location='cpu')
        self.dvae.load_state_dict(ckp['state_dict'])
        # fix dVAE
        for p in self.dvae.parameters():
            p.requires_grad = False
        self.dvae.eval()

    def _build_decoder(self):
        # Build Decoder
        # GPT-style causal masking Transformer decoder
        H, W = self.resolution
        self.h, self.w = H // self.down_factor, W // self.down_factor
        self.num_patches = self.h * self.w
        max_len = self.num_patches - 1
        self.trans_decoder = STEVETransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.dec_dict['dec_d_model'],
            n_head=self.dec_dict['dec_num_heads'],
            max_len=max_len,
            num_slots=self.num_slots,
            num_layers=self.dec_dict['dec_num_layers'],
        )

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']

    def encode(self, img, prev_slots=None):
        """Encode from img to slots."""
        B, T = img.shape[:2]
        img = img.flatten(0, 1)

        encoder_out = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # apply SlotAttn on video frames via reusing slots
        all_slots, all_masks = [], []
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]
            else:
                latents = self.predictor(prev_slots)  # [B, N, C]

            # SA to get `post_slots`
            slots, masks = self.slot_attention(encoder_out[:, idx], latents)
            all_slots.append(slots)
            all_masks.append(masks.unflatten(-1, self.visual_resolution))

            # next timestep
            prev_slots = slots

        # (B, T, self.num_slots, self.slot_size)
        slots = torch.stack(all_slots, dim=1)
        # (B, T, self.num_slots, H, W)
        masks = torch.stack(all_masks, dim=1).contiguous()

        # resize masks to the original resolution
        if not self.training and self.visual_resolution != self.resolution:
            with torch.no_grad():
                masks = masks.flatten(0, 2).unsqueeze(1)  # [BTN, 1, H, W]
                masks = F.interpolate(
                    masks,
                    self.resolution,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).unflatten(0, (B, T, self.num_slots))

        return slots, masks, encoder_out

    def forward(self, data_dict):
        """A wrapper for model forward.

        If the input video is too long in testing, we manually cut it.
        """
        img = data_dict['img']
        T = img.shape[1]
        if T <= self.clip_len or self.training:
            return self._forward(
                img, img_token_id=data_dict.get('token_id', None))

        # try to find the max len to input each time
        clip_len = T
        while True:
            try:
                _ = self._forward(img[:, :clip_len])
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # update `clip_len`
        self.clip_len = max(self.clip_len, clip_len)
        # no need to split
        if clip_len == T:
            return self._forward(
                img, img_token_id=data_dict.get('token_id', None))

        # split along temporal dim
        cat_dict = None
        prev_slots = None
        for clip_idx in range(0, T, clip_len):
            out_dict = self._forward(
                img[:, clip_idx:clip_idx + clip_len], prev_slots=prev_slots)
            # because this should be in test mode, we detach the outputs
            if cat_dict is None:
                cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
            else:
                for k, v in out_dict.items():
                    cat_dict[k].append(v.detach())
            prev_slots = cat_dict['post_slots'][-1][:, -1].detach().clone()
            del out_dict
            torch.cuda.empty_cache()
        cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
        return cat_dict

    def _forward(self, img, img_token_id=None, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            img_token_id: [B, T, h*w], pre-computed dVAE tokenized img ids
            prev_slots: [B, num_slots, slot_size] or None,
                the `post_slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        slots, masks, encoder_out = self.encode(img, prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]
        # `masks` has shape: [B, T, self.num_slots, H, W]

        out_dict = {'slots': slots, 'masks': masks}
        if self.testing:
            return out_dict

        # tokenize the images
        if img_token_id is None:
            with torch.no_grad():
                img_token_id = self.dvae.tokenize(
                    img, one_hot=False).flatten(2, 3).detach()
        h, w = self.h, self.w
        target_token_id = img_token_id.flatten(0, 1).long()  # [B*T, h*w]

        # TransformerDecoder token prediction loss
        in_slots = slots.flatten(0, 1)  # [B*T, N, C]
        in_token_id = target_token_id[:, :-1]
        pred_token_id = self.trans_decoder(in_slots, in_token_id)[:, -(h * w):]
        # [B*T, h*w, vocab_size]
        out_dict.update({
            'pred_token_id': pred_token_id,
            'target_token_id': target_token_id,
        })

        # decode image for loss computing
        if self.use_img_recon_loss:
            out_dict['gt_img'] = img.flatten(0, 1)  # [B*T, C, H, W]
            logits = pred_token_id.transpose(2, 1).\
                unflatten(-1, (self.h, self.w)).contiguous()  # [B*T, S, h, w]
            z_logits = F.log_softmax(logits, dim=1)
            z = gumbel_softmax(z_logits, tau=0.1, hard=False, dim=1)
            recon_img = self.dvae.detokenize(z)  # [B*T, C, H, W]
            out_dict['recon_img'] = recon_img

        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        pred_token_id = out_dict['pred_token_id'].flatten(0, 1)
        target_token_id = out_dict['target_token_id'].flatten(0, 1)
        token_recon_loss = F.cross_entropy(pred_token_id, target_token_id)
        loss_dict = {'token_recon_loss': token_recon_loss}
        if self.use_img_recon_loss:
            gt_img = out_dict['gt_img']
            recon_img = out_dict['recon_img']
            recon_loss = F.mse_loss(recon_img, gt_img)
            loss_dict['img_recon_loss'] = recon_loss
        return loss_dict
