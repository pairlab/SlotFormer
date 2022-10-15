import torch

from nerv.training import BaseModel

from .transformer import CLEVRERTransformerModel


class CLEVRERAloe(BaseModel):
    """VQA model on CLEVRER dataset using pre-computed img embeddings."""

    def __init__(self, transformer_model: CLEVRERTransformerModel):
        super().__init__()

        self.transformer_model = transformer_model

    def forward(self, data_dict):
        return self.transformer_model(data_dict)

    def calc_train_loss(self, data_dict, out_dict):
        loss_dict = self.transformer_model.loss_function(data_dict, out_dict)
        if out_dict['cls_answer_logits'] is None:
            cls_bs = 0
        else:
            cls_bs = out_dict['cls_answer_logits'].shape[0]
        if out_dict['mc_answer_logits'] is None:
            mc_bs = 0
        else:
            mc_bs = out_dict['mc_answer_logits'].shape[0]
        loss_dict['cls_bs'] = cls_bs
        loss_dict['mc_bs'] = mc_bs
        return loss_dict

    @torch.no_grad()
    def _eval_q_subtype(self, corr_ques, q_subtypes, subtype_id):
        """Calculate the per-question accuracy for a subtype of questions."""
        if corr_ques is None:
            return torch.tensor(0.).to(self.device), 0
        # `corr_ques` and `q_subtypes` are of shape [B]
        subtype_mask = (q_subtypes == subtype_id)
        if not subtype_mask.any():
            return torch.tensor(0.).to(self.device), 0
        bs = subtype_mask.sum().item()
        acc = corr_ques[subtype_mask].sum() / float(bs)
        return acc, bs

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval, we only care about QA accuracy."""
        cls_answer_logits = out_dict['cls_answer_logits']
        if cls_answer_logits is None:
            cls_acc = torch.tensor(0.).to(self.device)
            cls_bs = 0
        else:
            cls_labels = data_dict['cls_label'].long()
            cls_bs = cls_labels.shape[0]
            cls_preds = cls_answer_logits.argmax(-1)
            cls_acc = (cls_preds == cls_labels).float().sum() / \
                cls_labels.shape[0]

        mc_subtype = data_dict['mc_subtype']  # [B]
        mc_answer_logits = out_dict['mc_answer_logits']  # [Bn]
        if mc_answer_logits is not None:
            mc_labels = data_dict['mc_label']  # [Bn]
            mc_preds = (mc_answer_logits > 0.).type_as(mc_labels)
            mc_correct_mask = (mc_preds == mc_labels).float()
            mc_flag = data_dict['mc_flag']  # [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
            mc_bs = mc_flag.max().item() + 1
            mc_corr_ques = []
            for i in range(mc_bs):
                mc_corr_ques.append(mc_correct_mask[mc_flag == i].all().item())
            mc_corr_ques = torch.tensor(mc_corr_ques).to(self.device)
            mc_acc = mc_corr_ques.sum() / float(mc_bs)
        else:
            mc_corr_ques = None
            mc_acc, mc_bs = torch.tensor(0.).to(self.device), 0

        exp_acc, exp_bs = self._eval_q_subtype(mc_corr_ques, mc_subtype, 1)
        pred_acc, pred_bs = self._eval_q_subtype(mc_corr_ques, mc_subtype, 2)
        count_acc, count_bs = self._eval_q_subtype(mc_corr_ques, mc_subtype, 3)

        return {
            'descriptive_acc': cls_acc,
            'descriptive_bs': cls_bs,
            'multiple-choice_acc': mc_acc,
            'multiple-choice_bs': mc_bs,
            'explanatory_acc': exp_acc,
            'explanatory_bs': exp_bs,
            'predictive_acc': pred_acc,
            'predictive_bs': pred_bs,
            'counterfactual_acc': count_acc,
            'counterfactual_bs': count_bs,
        }

    @property
    def dtype(self):
        return self.transformer_model.CLS.dtype

    @property
    def device(self):
        return self.transformer_model.CLS.device
