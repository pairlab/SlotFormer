import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from nerv.training import BaseModel
from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

from .utils import assert_shape, SoftPositionEmbed, torch_cat
from .predictor import ResidualMLPPredictor, TransformerPredictor, \
    RNNPredictorWrapper


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.
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
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

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

        return slots

    @property
    def dtype(self):
        return self.project_k.weight.dtype

    @property
    def device(self):
        return self.project_k.weight.device


class StoSAVi(BaseModel):
    """SA model with stochastic kernel and additional prior_slots head.
    If loss_dict['kld_method'] = 'none', it becomes a standard SAVi model.
    """

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
            kernel_mlp=True,
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ),
        pred_dict=dict(
            pred_type='transformer',
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_post_recon_loss=True,
            kld_method='var-0.01',  # 'none' to make it deterministic
        ),
        eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps

        self.slot_dict = slot_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict
        self.loss_dict = loss_dict

        self._build_slot_attention()
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

        # predict the (\mu, \sigma) to sample the `kernels` input to SA
        if self.slot_dict.get('kernel_mlp', True):
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2),
                nn.LayerNorm(self.slot_size * 2),
                nn.ReLU(),
                nn.Linear(self.slot_size * 2, self.slot_size * 2),
            )
        else:
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2), )

        # predict the `prior_slots`
        # useless, just for compatibility to load pre-trained weights
        self.prior_slot_layer = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size),
            nn.LayerNorm(self.slot_size),
            nn.ReLU(),
            nn.Linear(self.slot_size, self.slot_size),
        )

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_encoder(self):
        # Build Encoder
        # Conv CNN --> PosEnc --> MLP
        self.enc_channels = list(self.enc_dict['enc_channels'])  # CNN channels
        self.enc_ks = self.enc_dict['enc_ks']  # kernel size in CNN
        self.enc_norm = self.enc_dict['enc_norm']  # norm in CNN
        self.visual_resolution = (64, 64)  # CNN out visual resolution
        self.visual_channels = self.enc_channels[-1]  # CNN out visual channels

        enc_layers = len(self.enc_channels) - 1
        self.encoder = nn.Sequential(*[
            conv_norm_act(
                self.enc_channels[i],
                self.enc_channels[i + 1],
                kernel_size=self.enc_ks,
                # 2x downsampling for 128x128 image
                stride=2 if (i == 0 and self.resolution[0] == 128) else 1,
                norm=self.enc_norm,
                act='relu' if i != (enc_layers - 1) else '')
            for i in range(enc_layers)
        ])  # relu except for the last layer

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(self.visual_channels,
                                                       self.visual_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )

    def _build_decoder(self):
        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        self.dec_channels = self.dec_dict['dec_channels']  # CNN channels
        self.dec_resolution = self.dec_dict['dec_resolution']  # broadcast size
        self.dec_ks = self.dec_dict['dec_ks']  # kernel size
        self.dec_norm = self.dec_dict['dec_norm']  # norm in CNN
        assert self.dec_channels[0] == self.slot_size, \
            'wrong in_channels for Decoder'
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            if out_size == self.resolution[0]:
                stride = 1
            modules.append(
                deconv_norm_act(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=self.dec_ks,
                    stride=stride,
                    norm=self.dec_norm,
                    act='relu'))
            out_size = deconv_out_shape(out_size, stride, self.dec_ks // 2,
                                        self.dec_ks, stride - 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1], 4, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.slot_size,
                                                       self.dec_resolution)

    def _build_predictor(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Predictor
        pred_type = self.pred_dict.get('pred_type', 'transformer')
        # Transformer (object interaction) --> LSTM (scene dynamic)
        if pred_type == 'mlp':
            self.predictor = ResidualMLPPredictor(
                [self.slot_size, self.slot_size * 2, self.slot_size],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        else:
            self.predictor = TransformerPredictor(
                self.slot_size,
                self.pred_dict['pred_num_layers'],
                self.pred_dict['pred_num_heads'],
                self.pred_dict['pred_ffn_dim'],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        # wrap LSTM
        if self.pred_dict['pred_rnn']:
            self.predictor = RNNPredictorWrapper(
                self.predictor,
                self.slot_size,
                self.slot_mlp_size,
                num_layers=1,
                rnn_cell='LSTM',
                sg_every=self.pred_dict['pred_sg_every'],
            )

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_post_recon_loss = self.loss_dict['use_post_recon_loss']
        assert self.use_post_recon_loss
        # stochastic SAVi by sampling the kernels
        kld_method = self.loss_dict['kld_method']
        # a smaller sigma for the prior distribution
        if '-' in kld_method:
            kld_method, kld_var = kld_method.split('-')
            self.kld_log_var = math.log(float(kld_var))
        else:
            self.kld_log_var = math.log(1.)
        self.kld_method = kld_method
        assert self.kld_method in ['var', 'none']

    def _kld_loss(self, prior_dist, post_slots):
        """KLD between (mu, sigma) and (0 or mu, 1)."""
        if self.kld_method == 'none':
            return torch.tensor(0.).type_as(prior_dist)
        assert prior_dist.shape[-1] == self.slot_size * 2
        mu1 = prior_dist[..., :self.slot_size]
        log_var1 = prior_dist[..., self.slot_size:]
        mu2 = mu1.detach().clone()  # no penalty for mu
        log_var2 = torch.ones_like(log_var1).detach() * self.kld_log_var
        sigma1 = torch.exp(log_var1 * 0.5)
        sigma2 = torch.exp(log_var2 * 0.5)
        kld = torch.log(sigma2 / sigma1) + \
            (torch.exp(log_var1) + (mu1 - mu2)**2) / \
            (2. * torch.exp(log_var2)) - 0.5
        return kld.sum(-1).mean()

    def _sample_dist(self, dist):
        """Sample values from Gaussian distribution."""
        assert dist.shape[-1] == self.slot_size * 2
        mu = dist[..., :self.slot_size]
        # not doing any stochastic
        if self.kld_method == 'none':
            return mu
        log_var = dist[..., self.slot_size:]
        eps = torch.randn_like(mu).detach()
        sigma = torch.exp(log_var * 0.5)
        return mu + eps * sigma

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        encoder_out = self.encoder(img).type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [B, C, H, W]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [B, C, H*W]
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out

    def encode(self, img, prev_slots=None):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]

        # apply SlotAttn on video frames via reusing slots
        all_kernel_dist, all_post_slots = [], []
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = init_latents  # [B, N, C]
            else:
                latents = self.predictor(prev_slots)  # [B, N, C]

            # stochastic `kernels` as SA input
            kernel_dist = self.kernel_dist_layer(latents)
            kernels = self._sample_dist(kernel_dist)
            all_kernel_dist.append(kernel_dist)

            # perform SA to get `post_slots`
            post_slots = self.slot_attention(encoder_out[:, idx], kernels)
            all_post_slots.append(post_slots)

            # next timestep
            prev_slots = post_slots

        # (B, T, self.num_slots, self.slot_size)
        kernel_dist = torch.stack(all_kernel_dist, dim=1)
        post_slots = torch.stack(all_post_slots, dim=1)

        return kernel_dist, post_slots, encoder_out

    def _reset_rnn(self):
        self.predictor.reset()

    def forward(self, data_dict):
        """A wrapper for model forward.

        If the input video is too long in testing, we manually cut it.
        """
        img = data_dict['img']
        T = img.shape[1]
        if T <= self.clip_len or self.training:
            return self._forward(img, None)

        # try to find the max len to input each time
        clip_len = T
        while True:
            try:
                _ = self._forward(img[:, :clip_len], None)
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # update `clip_len`
        self.clip_len = max(self.clip_len, clip_len)
        # no need to split
        if clip_len == T:
            return self._forward(img, None)

        # split along temporal dim
        cat_dict = None
        prev_slots = None
        for clip_idx in range(0, T, clip_len):
            out_dict = self._forward(img[:, clip_idx:clip_idx + clip_len],
                                     prev_slots)
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

    def _forward(self, img, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            prev_slots: [B, num_slots, slot_size] or None,
                the `post_slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        B, T = img.shape[:2]
        kernel_dist, post_slots, encoder_out = \
            self.encode(img, prev_slots=prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]

        out_dict = {
            'post_slots': post_slots,  # [B, T, num_slots, C]
            'kernel_dist': kernel_dist,  # [B, T, num_slots, 2C]
            'img': img,  # [B, T, 3, H, W]
        }
        if self.testing:
            return out_dict

        if self.use_post_recon_loss:
            post_recon_img, post_recons, post_masks, _ = \
                self.decode(post_slots.flatten(0, 1))
            post_dict = {
                'post_recon_combined': post_recon_img,  # [B*T, 3, H, W]
                'post_recons': post_recons,  # [B*T, num_slots, 3, H, W]
                'post_masks': post_masks,  # [B*T, num_slots, 1, H, W]
            }
            out_dict.update(
                {k: v.unflatten(0, (B, T))
                 for k, v in post_dict.items()})

        return out_dict

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution
        num_channels = 3

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].

        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        kld_loss = self._kld_loss(out_dict['kernel_dist'],
                                  out_dict['post_slots'])
        loss_dict = {
            'kld_loss': kld_loss,
        }
        if self.use_post_recon_loss:
            post_recon_combined = out_dict['post_recon_combined']
            img = out_dict['img']
            loss_dict['post_recon_loss'] = F.mse_loss(post_recon_combined, img)
        return loss_dict

    @property
    def dtype(self):
        return self.slot_attention.dtype

    @property
    def device(self):
        return self.slot_attention.device
