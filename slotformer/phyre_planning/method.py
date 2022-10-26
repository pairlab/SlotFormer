import wandb
import numpy as np

import torch
from torch.utils.data._utils.collate import default_collate

from slotformer.base_slots.method import to_rgb_from_tensor, SlotBaseMethod


def build_method(**kwargs):
    params = kwargs['params']
    assert params.model == 'PHYREReadout', \
        f'{params.model} method is not implemented.'
    return PHYREReadoutMethod(**kwargs)


class PHYREReadoutMethod(SlotBaseMethod):
    """PHYREReadout model training method."""

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        save_load_img = dst.load_img
        dst.load_img = True
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst).numpy()
        batch = default_collate([dst.__getitem__(i) for i in sampled_idx])
        batch = {k: v.cuda() for k, v in batch.items()}
        dst.load_img = save_load_img
        out_dict = model(batch)

        videos = to_rgb_from_tensor(batch['img'])  # [B, T, C, H, W]
        videos = (videos * 255.).cpu().numpy().astype(np.uint8)
        gts = batch['label'].flatten()  # [B]
        preds = torch.sigmoid(out_dict['logits'].flatten())  # [B]
        texts = [
            f'GT: {int(gt.item())}. Pred: {pred.item():.3f}'
            for gt, pred in zip(gts, preds)
        ]
        log_dict = {
            f'val/video{i}': wandb.Video(videos[i], fps=8, caption=texts[i])
            for i in range(videos.shape[0])
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
