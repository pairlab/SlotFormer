import wandb
import numpy as np

import torch

from nerv.utils import AverageMeter, MeanMetric
from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

from slotformer.base_slots.models import to_rgb_from_tensor

from .datasets import clevrer_collate_fn


def build_method(**kwargs):
    params = kwargs['params']
    assert params.model == 'CLEVRERAloe', \
        f'{params.model} method is not implemented.'
    return CLEVRERAloeMethod(**kwargs)


class CLEVRERAloeMethod(BaseMethod):
    """General method class for CLEVRER VQA training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert not self.params.loss_dict['use_mask_obj_loss'], \
            "don't use `mask_obj_loss` when using SAVi slots"

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        optimizer = super()._configure_optimizers()[0]

        lr = self.params.lr
        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # this visualization is not very helpful
        # uncomment if you want to see it
        # self._sample_qa_pairs(model)
        return

    @torch.no_grad()
    def _accumulate_stats(self, stats_dict, test=False):
        """Append stats in `stats_dict` to `self.stats_dict`.

        On CLEVRER, we have two types of questions, so need to accumulate them
            separately, i.e. by two diff batch_size.
        """
        if not test:
            bs = stats_dict.pop('batch_size', 1)
            cls_bs = stats_dict.pop('cls_bs', 0)
            mc_bs = stats_dict.pop('mc_bs', 0)
            if self.stats_dict is None:
                self.stats_dict = {
                    k: AverageMeter(device=self.device)
                    for k in stats_dict.keys()
                }

            if cls_bs > 0:
                self.stats_dict['cls_answer_loss'].update(
                    stats_dict['cls_answer_loss'].item(), cls_bs)
            if mc_bs > 0:
                self.stats_dict['mc_answer_loss'].update(
                    stats_dict['mc_answer_loss'].item(), mc_bs)
            self.stats_dict['loss'].avg = \
                self.stats_dict['cls_answer_loss'].avg + \
                self.stats_dict['mc_answer_loss'].avg

            if 'mask_obj_loss' in stats_dict:
                self.stats_dict['mask_obj_loss'].update(
                    stats_dict['mask_obj_loss'].item(), bs)
                self.stats_dict['loss'].avg += \
                    self.stats_dict['mask_obj_loss'].avg
            return

        # eval
        all_q = [
            'descriptive', 'multiple-choice', 'explanatory', 'predictive',
            'counterfactual'
        ]
        _ = stats_dict.pop('batch_size', 1)
        if self.stats_dict is None:
            self.stats_dict = {
                f'{q}_acc': MeanMetric(device=self.device)
                for q in all_q
            }
        for q in all_q:
            bs = stats_dict.pop(f'{q}_bs', 0)
            acc = stats_dict.pop(f'{q}_acc', 0)
            if bs > 0:
                item = self._make_tensor(acc.item())
                self.stats_dict[f'{q}_acc'].update(item, bs)

    @torch.no_grad()
    def _sample_qa_pairs(self, model):
        """Visualize some results."""
        model.eval()
        val_set = self.val_loader.dataset
        ori_flag = val_set.load_frames
        val_set.load_frames = True
        n_samples = self.params.n_samples
        cls_sample = n_samples // 2
        mc_sample = n_samples - cls_sample
        cls_idx = np.random.choice(
            range(val_set.num_cls_questions), (cls_sample, ), replace=False)
        mc_idx = np.random.choice(
            range(val_set.num_cls_questions, len(val_set)), (mc_sample, ),
            replace=False)
        idx = np.concatenate([cls_idx, mc_idx])
        batch = clevrer_collate_fn([val_set[i] for i in idx])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out_dict = model(batch)

        # deal with cls_questions, [n1, classes]
        cls_answer_logits = out_dict['cls_answer_logits']
        cls_answer_probs = torch.softmax(cls_answer_logits, dim=-1)
        top5_scores, top5_labels = torch.topk(cls_answer_probs, k=5, dim=-1)
        top5_answers = val_set.get_answer_from_label(top5_labels.cpu().numpy())
        top5_scores = top5_scores.cpu().numpy()
        pred_texts = [
            '    \n'.join([
                f'{top5_answers[i, j]}: {top5_scores[i, j]:.4f}'
                for j in range(5)
            ]) for i in range(cls_sample)
        ]
        # GT QA
        qa_texts = [val_set.get_qa_text(i) for i in idx]  # (Q, A)
        qa_texts = [f'Q: {pair[0]}  A: {pair[1]}' for pair in qa_texts]
        # video, [n1, T, C, H, W]
        cls_videos = to_rgb_from_tensor(batch['cls_video']).cpu().numpy()
        cls_videos = np.round(cls_videos * 255.).astype(np.uint8)
        # log videos, GT QA and predicted answers
        log_dict = {
            f'val/video{i}': wandb.Video(
                cls_videos[i],
                caption=f'{qa_texts[i]}\n{pred_texts[i]}',
                fps=5)
            for i in range(cls_sample)
        }
        wandb.log(log_dict, step=self.it, commit=False)

        # deal with mc_questions, [Bn]
        mc_answer_logits = out_dict['mc_answer_logits']
        mc_answer_probs = torch.sigmoid(mc_answer_logits)
        mc_answers = (mc_answer_probs > 0.5).detach().cpu().numpy()
        mc_flag = batch['mc_flag'].cpu().numpy()  # [0, 0, 0, 1, 1, 1, 1, 2, 2]
        # GT QCA
        qca_texts = [val_set.get_qa_text(i) for i in mc_idx]  # (Q, choices, A)
        q_texts = [f'Q: {pair[0]}\n' for pair in qca_texts]
        ca_texts = []
        for i in range(mc_sample):
            num_choices = (mc_flag == i).sum()
            ca_texts.append('    \n'.join([
                f'C: {qca_texts[i][1][j]}  A: {qca_texts[i][2][j]}  '
                f'Pred: {mc_answers[mc_flag == i][j]}'
                for j in range(num_choices)
            ]))
        # video, [B, T, C, H, W]
        mc_videos = to_rgb_from_tensor(batch['mc_video']).cpu().numpy()
        mc_videos = np.round(mc_videos * 255.).astype(np.uint8)
        # log videos, GT QCA and predicted answers
        log_dict = {
            f'val/video{i + cls_sample}': wandb.Video(
                mc_videos[i], caption=f'{q_texts[i]}\n{ca_texts[i]}', fps=5)
            for i in range(mc_sample)
        }
        wandb.log(log_dict, step=self.it)
        val_set.load_frames = ori_flag
