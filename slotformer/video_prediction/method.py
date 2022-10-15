import os
import wandb
import numpy as np

import torch
import torchvision.utils as vutils

from nerv.training import BaseMethod

from slotformer.base_slots.method import SAViMethod, STEVEMethod, \
    to_rgb_from_tensor


def build_method(**kwargs):
    params = kwargs['params']
    if params.model in ['SlotFormer', 'SingleStepSlotFormer']:
        return SlotFormerMethod(**kwargs)
    elif params.model == 'STEVESlotFormer':
        return STEVESlotFormerMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class SlotFormerMethod(SAViMethod):

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        if not hasattr(self.params, 'use_loss_decay'):
            return

        # decay the temporal weighting linearly
        if not self.params.use_loss_decay:
            self.model.module.loss_decay_factor = 1.
            return

        cur_steps = self.it
        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.loss_decay_pct * total_steps

        if cur_steps >= decay_steps:
            self.model.module.loss_decay_factor = 1.
            return

        # increase tau linearly from 0.01 to 1
        self.model.module.loss_decay_factor = \
            0.01 + cur_steps / decay_steps * 0.99

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        if not hasattr(self.params, 'use_loss_decay'):
            return

        # also log the loss_decay_factor
        x = self.model.module.loss_decay_factor
        wandb.log({'train/loss_decay_factor': x}, step=self.it)

    def _compare_videos(self, img, recon_combined, rollout_combined):
        """Stack 3 videos to compare them."""
        # pause the 1st frame if on PHYRE
        if 'phyre' in self.params.dataset.lower():
            img, recon_combined, rollout_combined = [
                self._pause_frame(x)
                for x in [img, recon_combined, rollout_combined]
            ]
        # pad to the length of rollout video
        T = rollout_combined.shape[0]
        img = self._pad_frame(img, T)
        recon_combined = self._pad_frame(recon_combined, T)
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    img,  # original images
                    recon_combined,  # reconstructions
                    rollout_combined,  # rollouts
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                # pad white if using black background
                pad_value=1 if self.params.get('reverse_color', False) else 0,
            ) for i in range(img.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    def _read_video_and_slots(self, dst, idx):
        """Read the video and slots from the dataset."""
        # PHYRE
        if 'phyre' in self.params.dataset.lower():
            # read video
            data_dict = dst.get_video(idx, video_len=self.params.video_len)
            video = data_dict['video']
            # read slots
            slots = dst._read_slots(
                data_dict['data_idx'],
                video_len=self.params.video_len,
            )['slots']  # [T, N, C]
            slots = torch.from_numpy(slots).float().to(self.device)
        # OBJ3D, CLEVRER, Physion
        else:
            # read video
            video = dst.get_video(idx)['video']
            # read slots
            video_path = dst.files[idx]
            slots = dst.video_slots[os.path.basename(video_path)]  # [T, N, C]
            if self.params.frame_offset > 1:
                slots = np.ascontiguousarray(slots[::self.params.frame_offset])
            slots = torch.from_numpy(slots).float().to(self.device)
        T = min(video.shape[0], slots.shape[0])
        # video: [T, 3, H, W], slots: [T, N, C]
        return video[:T], slots[:T]

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        save_use_img_recon_loss = model.use_img_recon_loss
        model.use_img_recon_loss = sample_video  # do img_recon_loss eval
        save_loss_decay_factor = model.loss_decay_factor
        model.loss_decay_factor = 1.  # compute loss in normal scale
        BaseMethod.validation_epoch(self, model, san_check_step=san_check_step)
        model.use_img_recon_loss = save_use_img_recon_loss
        model.loss_decay_factor = save_loss_decay_factor

        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, rollout_results, compare_results = [], [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            T = video.shape[0]
            # reconstruct gt_slots as sanity-check
            # i.e. if the pre-trained weights are loaded correctly
            recon_combined, recons, masks, _ = model.decode(slots)
            img = video.type_as(recon_combined)
            save_video = self._make_video_grid(img, recon_combined, recons,
                                               masks)
            results.append(save_video)
            # rollout
            past_steps = self.params.input_frames
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            out_dict = model.rollout(
                past_slots, T - past_steps, decode=True, with_gt=True)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            rollout_combined, recons, masks = out_dict['recon_combined'], \
                out_dict['recons'], out_dict['masks']
            img = video.type_as(rollout_combined)
            pred_video = self._make_video_grid(img, rollout_combined, recons,
                                               masks)
            rollout_results.append(pred_video)  # per-slot rollout results
            # stack (gt video, gt slots recon video, slot_0 rollout video)
            # horizontally to better compare the 3 videos
            compare_video = self._compare_videos(img, recon_combined,
                                                 rollout_combined)
            compare_results.append(compare_video)

        log_dict = {
            'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
            'val/compare_video': self._convert_video(compare_results),
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()


class STEVESlotFormerMethod(SlotFormerMethod):

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # STEVE's Transformer-based decoder autoregressively reconstructs the
        # video, which is super slow
        # therefore, we don't perform visualization in eval
        # change this if you want to see reconstruction anyways
        recon_video = False
        super().validation_epoch(
            model, san_check_step=san_check_step, sample_video=recon_video)

    @staticmethod
    def _make_video(video, soft_video, hard_video):
        """Compare the 3 videos."""
        return STEVEMethod._make_video(video, soft_video, hard_video)

    def _slots2video(self, model, slots):
        """Decode slots to videos."""
        T = slots.shape[0]
        all_soft_recon, all_hard_recon, bs = [], [], 16  # to avoid OOM
        for idx in range(0, T, bs):
            soft_recon, hard_recon = model.decode(slots[idx:idx + bs])
            all_soft_recon.append(soft_recon.cpu())
            all_hard_recon.append(hard_recon.cpu())
            del soft_recon, hard_recon
            torch.cuda.empty_cache()
        soft_recon = torch.cat(all_soft_recon, dim=0)
        hard_recon = torch.cat(all_hard_recon, dim=0)
        return soft_recon, hard_recon

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, rollout_results = [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            T = video.shape[0]
            # recon as sanity-check
            soft_recon, hard_recon = self._slots2video(model, slots)
            save_video = self._make_video(video, soft_recon, hard_recon)
            results.append(save_video)
            # rollout
            past_steps = T // 4  # self.params.roll_history_len
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            pred_slots = model.rollout(past_slots, T - past_steps)[0]
            slots = torch.cat([slots[:past_steps], pred_slots], dim=0)
            soft_recon, hard_recon = self._slots2video(model, slots)
            save_video = self._make_video(
                video, soft_recon, hard_recon, history_len=past_steps)
            rollout_results.append(save_video)
            del soft_recon, hard_recon
            torch.cuda.empty_cache()

        log_dict = {
            'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
