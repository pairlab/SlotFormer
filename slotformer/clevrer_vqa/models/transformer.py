import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.utils import batch_cat_vec, batch_gather
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


def mask_v_embedding(v_embedding, mask_token):
    """Mask one position per frame."""
    B, T, N, C = v_embedding.shape
    mask_idx = torch.randint(0, N, (B * T, )).to(v_embedding.device)
    batch_idx = torch.arange(B * T).type_as(mask_idx)
    gt_v_emb = batch_gather(v_embedding.view(-1, N, C), mask_idx).detach()
    v_embedding.view(-1, N, C)[batch_idx, mask_idx] = \
        mask_token.repeat(B * T, 1)
    return v_embedding, mask_idx, gt_v_emb


class CLEVRERTransformerModel(nn.Module):
    """Model from Ding et al. 2020 (https://arxiv.org/abs/2012.08508)."""

    def __init__(
            self,
            transformer_dict=dict(
                input_len=207,
                input_dim=16,
                pos_enc='learnable',
                num_layers=28,
                num_heads=10,
                ffn_dim=1024,
                norm_first=True,
                cls_mlp_size=128,
            ),
            lang_dict=dict(
                question_len=20,
                question_vocab_size=82,
                answer_vocab_size=22,
            ),
            vision_dict=dict(vision_dim=64, ),
            loss_dict=dict(use_mask_obj_loss=False, ),
    ):
        super().__init__()

        # build transformer encoder
        input_dim = transformer_dict['input_dim']
        lang_emb_dim = input_dim - 2
        self.input_dim = input_dim + 2

        # concat id with text/vision tokens as Transformer input
        # from Aloe, tokens are fixed
        self.text_token = nn.Parameter(
            torch.tensor([1, 0]).float(), requires_grad=False)
        self.vision_token = nn.Parameter(
            torch.tensor([0, 1]).float(), requires_grad=False)
        # also question/choice need different ids
        self.cls_token = nn.Parameter(
            torch.tensor([0, 1]).float(), requires_grad=False)
        self.mc_question_token = nn.Parameter(
            torch.tensor([1, 0]).float(), requires_grad=False)
        self.mc_choice_token = nn.Parameter(
            torch.tensor([0, 1]).float(), requires_grad=False)

        num_trans_heads = transformer_dict['num_heads']
        self.d_model = self.input_dim * num_trans_heads  # from Aloe
        self.input_len = transformer_dict['input_len'] + 1  # CLS token
        self.transformer_encoder = build_transformer(
            self.input_len,
            transformer_dict['pos_enc'],
            self.d_model,
            num_trans_heads,
            transformer_dict['ffn_dim'],
            transformer_dict['num_layers'],
            norm_first=transformer_dict['norm_first'],
        )

        # build language related modules
        self.q_embedding = nn.Embedding(lang_dict['question_vocab_size'],
                                        lang_emb_dim)
        self.q_in_proj = nn.Linear(self.input_dim, self.d_model)
        self.question_len = lang_dict['question_len']

        self.num_answer_classes = lang_dict['answer_vocab_size']
        cls_mlp_size = transformer_dict['cls_mlp_size']
        self.cls_answer_mlp = nn.Sequential(
            nn.Linear(self.d_model, cls_mlp_size),
            nn.ReLU(),
            nn.Linear(cls_mlp_size, self.num_answer_classes),
        )
        self.mc_answer_mlp = nn.Sequential(
            nn.Linear(self.d_model, cls_mlp_size),
            nn.ReLU(),
            nn.Linear(cls_mlp_size, 1),
        )

        # build vision related modules
        vision_dim = vision_dict['vision_dim'] + 2
        self.vision_in_proj = nn.Linear(vision_dim, self.d_model)

        # learnable [CLS] token
        # from Aloe, CLS token is zero inited
        self.CLS = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # whether to use masked object prediction loss
        self.mask_obj_loss = loss_dict['use_mask_obj_loss']
        if self.mask_obj_loss:
            # learnable [MASK] token
            self.mask_token = nn.Parameter(torch.zeros((1, vision_dim - 2)))
            self.mask_obj_fc = nn.Linear(self.d_model, vision_dim - 2)
        assert not self.mask_obj_loss, \
            "don't use `mask_obj_loss` when using SAVi slots"

    def _process_in_embeddings(self, v_embedding, q_embedding, q_pad_mask):
        """Prepare input for Transformer.

        Args:
            v_embedding: [B, T, N, C1]
            q_embedding: [B, L, C2]
            q_pad_mask: [B, L]

        Returns:
            in_embedding: [B, in_len, C] (in_len = 1 + T*N + L)
            pad_mask: [B, in_len]
            mask_idx: [B * T]
            gt_v_emb: [B * T, C1], one emb per frame
        """
        bs = q_embedding.shape[0]

        # masked object prediction loss
        # according to Aloe, we mask one object per timestep
        if self.mask_obj_loss:
            v_embedding, mask_idx, gt_v_emb = mask_v_embedding(
                v_embedding, self.mask_token)
        else:
            mask_idx, gt_v_emb = None, None

        # unroll along temporal dim
        v_embedding = v_embedding.flatten(1, 2)  # [B, T*N, C2]
        v_embedding = batch_cat_vec(v_embedding, self.vision_token, dim=-1)
        v_embedding = self.vision_in_proj(v_embedding)  # [B, T*N, C]

        q_embedding = batch_cat_vec(q_embedding, self.text_token, dim=-1)
        q_embedding = self.q_in_proj(q_embedding)  # [B, L, C]

        CLS = self.CLS.repeat(bs, 1, 1)
        in_embedding = torch.cat([CLS, v_embedding, q_embedding], dim=1)

        # construct padding mask, CLS and vision tokens should be False
        no_pad_mask = torch.zeros(bs, self.input_len -
                                  q_pad_mask.shape[1]).type_as(q_pad_mask)
        pad_mask = torch.cat([no_pad_mask, q_pad_mask], dim=-1)
        return in_embedding, pad_mask, mask_idx, gt_v_emb

    def _cls_forward(self, inputs):
        """Apply model to CLEVRER cls (descriptive) questions.

        Args:
            inputs (dict): dict with keys:
                - cls_video_emb: [B, T, N, C1]
                - cls_q_tokens: [B, L]
                - cls_q_pad_mask: [B, L]
        """
        # no cls question in this batch
        if len(inputs['cls_q_tokens']) == 0:
            return None

        v_embedding = inputs['cls_video_emb']  # [B, T, N, C1]
        B, T, N, _ = v_embedding.shape

        q_tokens, q_pad_mask = inputs['cls_q_tokens'], inputs['cls_q_pad_mask']
        q_embedding = self.q_embedding(q_tokens)  # [B, L, C2]
        q_embedding = batch_cat_vec(q_embedding, self.cls_token, dim=-1)

        in_embedding, pad_mask, mask_idx, gt_v_emb = \
            self._process_in_embeddings(v_embedding, q_embedding, q_pad_mask)

        # apply Transformer, [B, 1 + T*N + L, d_model]
        transformer_out = self.transformer_encoder(
            in_embedding, src_key_padding_mask=pad_mask)

        # multi-class classification
        cls_emb = transformer_out[:, 0, :]
        answer_logits = self.cls_answer_mlp(cls_emb)  # [B, num_classes]

        # masked object prediction
        if self.mask_obj_loss:
            out_v_emb = transformer_out[:, 1:1 + T * N].reshape(B * T, N, -1)
            mask_v_emb = batch_gather(out_v_emb, mask_idx)
            pred_v_emb = self.mask_obj_fc(mask_v_emb)  # [B * T, C1]
        else:
            pred_v_emb = None

        return {
            'answer_logits': answer_logits,
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }

    def _mc_forward(self, inputs):
        """Apply model to CLEVRER mc (multiple choice) questions.

        Args:
            inputs (dict): dict with keys:
                - mc_video_emb: [B, T, N, C1]
                - mc_q_tokens: [Bn, L]
                - mc_q_pad_mask: [Bn, L]
                - mc_flag: [Bn], e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
                    indicating which first_dim_idx corresponds to which video
        """
        # no mc question in this batch
        if len(inputs['mc_q_tokens']) == 0:
            return None

        # repeat v_embedding to pair up with each question
        v_embedding = inputs['mc_video_emb']  # [B, T, N, C1]
        mc_flag = inputs['mc_flag']  # [Bn]
        v_embedding = v_embedding[mc_flag.long()]  # [Bn, T, N, C1]
        B, T, N, _ = v_embedding.shape

        # need to split question and choice text
        q_tokens, q_pad_mask = inputs['mc_q_tokens'], inputs['mc_q_pad_mask']
        q_embedding = self.q_embedding(q_tokens)  # [Bn, L, C2]
        question = q_embedding[:, :self.question_len]
        choice = q_embedding[:, self.question_len:]
        q_embedding = torch.cat([
            batch_cat_vec(question, self.mc_question_token, dim=-1),
            batch_cat_vec(choice, self.mc_choice_token, dim=-1),
        ], 1)  # [Bn, L, C2']

        in_embedding, pad_mask, mask_idx, gt_v_emb = \
            self._process_in_embeddings(v_embedding, q_embedding, q_pad_mask)

        # apply Transformer, [Bn, 1 + T*N + L, d_model]
        transformer_out = self.transformer_encoder(
            in_embedding, src_key_padding_mask=pad_mask)

        # binary classification
        cls_emb = transformer_out[:, 0, :]  # [Bn, C]
        answer_logits = self.mc_answer_mlp(cls_emb)  # [Bn, 1]

        # masked object prediction
        if self.mask_obj_loss:
            out_v_emb = transformer_out[:, 1:1 + T * N].reshape(B * T, N, -1)
            mask_v_emb = batch_gather(out_v_emb, mask_idx)
            pred_v_emb = self.mask_obj_fc(mask_v_emb)  # [B * T, C1]
        else:
            pred_v_emb = None

        return {
            'answer_logits': answer_logits.view(-1),
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }

    def forward(self, inputs):
        """Applies model to CLEVRER questions.

        Args:
            inputs (dict): with keys:
                - cls_/mc_video (optional): [B, T, C, H, W]
                - cls_/mc_video_emb (optional): [B, T, N, C]

                - cls_q_tokens: [B, L]
                - cls_q_pad_mask: [B, L]
                - cls_label: [B]

                - mc_subtype: [B]
                - mc_q_tokens: [Bn, L]
                - mc_q_pad_mask: [Bn, L]
                - mc_label: [Bn]
                - mc_flag: [Bn], e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
                    indicating which first_dim_idx corresponds to which video

        Returns:
            torch.Tensor: [B, num_cls/num_choices], predicted answer logits
        """
        cls_dict = self._cls_forward(inputs)
        mc_dict = self._mc_forward(inputs)
        cls_answer_logits = cls_dict['answer_logits'] if \
            cls_dict is not None else None
        mc_answer_logits = mc_dict['answer_logits'] if \
            mc_dict is not None else None
        if self.mask_obj_loss:
            gt_v_emb = torch.cat(
                [d['gt_v_emb'] for d in [cls_dict, mc_dict] if d is not None],
                0)
            pred_v_emb = torch.cat([
                d['pred_v_emb'] for d in [cls_dict, mc_dict] if d is not None
            ], 0)
        else:
            gt_v_emb, pred_v_emb = None, None

        return {
            'cls_answer_logits': cls_answer_logits,  # [B1, num_cls]
            'mc_answer_logits': mc_answer_logits,  # [B2 n]
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }

    def loss_function(self, data_dict, out_dict):
        """Calculate VQA loss.

        Args:
            data_dict (dict): a dict with keys:
                - cls_label (torch.Tensor): [B]
                - mc_label (torch.Tensor): [Bn]
            out_dict (dict): a dict with keys:
                - cls_answer_logits (torch.Tensor): [B, num_classes]
                - mc_answer_logits (torch.Tensor): [Bn]
        """
        cls_logits = out_dict['cls_answer_logits']
        if cls_logits is None:
            cls_loss = torch.tensor(0.).type_as(self.CLS)
        else:
            cls_labels = data_dict['cls_label'].long()
            cls_loss = F.cross_entropy(cls_logits, cls_labels)

        mc_logits = out_dict['mc_answer_logits']
        if mc_logits is None:
            mc_loss = torch.tensor(0.).type_as(self.CLS)
        else:
            mc_labels = data_dict['mc_label'].type_as(mc_logits)
            mc_loss = F.binary_cross_entropy_with_logits(mc_logits, mc_labels)

        loss_dict = {
            'cls_answer_loss': cls_loss,
            'mc_answer_loss': mc_loss,
        }

        # self-supervised loss
        if self.mask_obj_loss:
            loss_dict['mask_obj_loss'] = F.mse_loss(out_dict['pred_v_emb'],
                                                    out_dict['gt_v_emb'])

        return loss_dict
