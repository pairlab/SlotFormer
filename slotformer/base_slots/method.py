import wandb
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

from .models import cosine_anneal, get_lr, gumbel_softmax, make_one_hot, \
    to_rgb_from_tensor


def build_method(**kwargs):
    params = kwargs['params']
    if params.model == 'StoSAVi':
        return SAViMethod(**kwargs)
    elif params.model == 'dVAE':
        return dVAEMethod(**kwargs)
    elif params.model == 'STEVE':
        return STEVEMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class SlotBaseMethod(BaseMethod):
    """Base method in this project."""

    @staticmethod
    def _pad_frame(video, target_T):
        """Pad the video to a target length at the end"""
        if video.shape[0] >= target_T:
            return video
        dup_video = torch.stack(
            [video[-1]] * (target_T - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)

    @staticmethod
    def _pause_frame(video, N=4):
        """Pause the video on the first frame by duplicating it"""
        dup_video = torch.stack([video[0]] * N, dim=0)
        return torch.cat([dup_video, video], dim=0)

    def _convert_video(self, video, caption=None):
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).numpy().astype(np.uint8)
        return wandb.Video(video, fps=self.vis_fps, caption=caption)

    @staticmethod
    def _get_sample_idx(N, dst):
        """Load videos uniformly from the dataset."""
        dst_len = len(dst.files)  # treat each video as a sample
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

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

    @property
    def vis_fps(self):
        # PHYRE
        if 'phyre' in self.params.dataset.lower():
            return 4
        # OBJ3D, CLEVRER, Physion
        else:
            return 8


class SAViMethod(SlotBaseMethod):
    """SAVi model training method."""

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # pause the video on the 1st frame in PHYRE
        if 'phyre' in self.params.dataset.lower():
            imgs, recon_combined, recons, masks = [
                self._pause_frame(x)
                for x in [imgs, recon_combined, recons, masks]
            ]
        # in PHYRE if the background is black, we scale the mask differently
        scale = 0. if self.params.get('reverse_color', False) else 1.
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1. - masks) * scale,  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=1. - scale,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, labels = [], []
        for i in sampled_idx:
            data_dict = dst.get_video(i.item())
            video, label = data_dict['video'].float().to(self.device), \
                data_dict.get('label', None)  # label for PHYRE
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            recon_combined, recons, masks = out_dict['post_recon_combined'], \
                out_dict['post_recons'], out_dict['post_masks']
            imgs = video.type_as(recon_combined)
            save_video = self._make_video_grid(imgs, recon_combined, recons,
                                               masks)
            results.append(save_video)
            labels.append(label)

        if all(lbl is not None for lbl in labels):
            caption = '\n'.join(
                ['Success' if lbl == 1 else 'Fail' for lbl in labels])
        else:
            caption = None
        wandb.log({'val/video': self._convert_video(results, caption=caption)},
                  step=self.it)
        torch.cuda.empty_cache()


class dVAEMethod(SlotBaseMethod):
    """dVAE model training method."""

    @staticmethod
    def _make_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(torch.stack([video, pred_video],
                                             dim=1))  # [T, 2, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 2*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results = []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            all_recons, bs = [], 100  # a hack to avoid OOM
            for batch_idx in range(0, video.shape[0], bs):
                data_dict = {
                    'img': video[batch_idx:batch_idx + bs],
                    'tau': 1.,
                    'hard': True,
                }
                recon = model(data_dict)['recon']
                all_recons.append(recon)
                torch.cuda.empty_cache()
            recon_video = torch.cat(all_recons, dim=0)
            save_video = self._make_video(video, recon_video)
            results.append(save_video)

        wandb.log({'val/video': self._convert_video(results)}, step=self.it)
        torch.cuda.empty_cache()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.tau_decay_pct * total_steps

        # decay tau
        self.model.module.tau = cosine_anneal(
            self.it,
            start_value=self.params.init_tau,
            final_value=self.params.final_tau,
            start_step=0,
            final_step=decay_steps,
        )

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # also log the tau
        wandb.log({'train/gumbel_tau': self.model.module.tau}, step=self.it)


class STEVEMethod(SlotBaseMethod):
    """STEVE model training method."""

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        assert self.params.optimizer.lower() == 'adam'
        assert self.params.weight_decay <= 0.
        lr = self.params.lr
        dec_lr = self.params.dec_lr

        # STEVE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
            },
        ]

        optimizer = optim.Adam(params_list, lr=lr, weight_decay=0.)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # STEVE's Transformer-based decoder autoregressively reconstructs the
        # video, which is super slow
        # therefore, we only visualize scene decomposition results
        # but don't show the video reconstruction
        # change this if you want to see reconstruction anyways
        self.recon_video = False
        super().validation_epoch(model, san_check_step=san_check_step)

    @staticmethod
    def _make_video(video, soft_video, hard_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    video.cpu(),  # original video
                    soft_video.cpu(),  # dVAE gumbel softmax reconstruction
                    hard_video.cpu(),  # argmax token reconstruction
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    @staticmethod
    def _make_slots_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video,  # [T, num_slots, 3, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        num_patches = model.num_patches
        n = int(num_patches**0.5)
        results, recon_results = [], []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            data_dict = {'img': video[None]}
            out_dict = model(data_dict)
            masks = out_dict['masks'][0]  # [T, num_slots, H, W]
            masked_video = video.unsqueeze(1) * masks.unsqueeze(2)
            # [T, num_slots, C, H, W]
            save_video = self._make_slots_video(video, masked_video)
            results.append(save_video)
            if not self.recon_video:
                continue

            # reconstruct the video by autoregressively generating patch tokens
            # using Transformer decoder conditioned on slots
            slots = out_dict['slots'][0]  # [T, num_slots, slot_size]
            all_soft_video, all_hard_video, bs = [], [], 16  # to avoid OOM
            for batch_idx in range(0, slots.shape[0], bs):
                _, logits = model.trans_decoder.generate(
                    slots[batch_idx:batch_idx + bs],
                    steps=num_patches,
                    sample=False,
                )
                # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
                logits = logits.transpose(2, 1).unflatten(
                    -1, (n, n)).contiguous().cuda()
                # 1. use logits after gumbel softmax to reconstruct the video
                z_logits = F.log_softmax(logits, dim=1)
                z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
                recon_video = model.dvae.detokenize(z)
                all_soft_video.append(recon_video.cpu())
                del z_logits, z, recon_video
                torch.cuda.empty_cache()
                # 2. SLATE directly use ont-hot token (argmax) as input
                z_hard = make_one_hot(logits, dim=1)
                recon_video_hard = model.dvae.detokenize(z_hard)
                all_hard_video.append(recon_video_hard.cpu())
                del logits, z_hard, recon_video_hard
                torch.cuda.empty_cache()

            recon_video = torch.cat(all_soft_video, dim=0)
            recon_video_hard = torch.cat(all_hard_video, dim=0)
            save_video = self._make_video(video, recon_video, recon_video_hard)
            recon_results.append(save_video)
            torch.cuda.empty_cache()

        log_dict = {'val/video': self._convert_video(results)}
        if self.recon_video:
            log_dict['val/recon_video'] = self._convert_video(recon_results)
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)
