"""Testing script for the video prediction task."""

import os
import sys
import numpy as np
import importlib
import argparse
from tqdm import tqdm

import torch

from nerv.utils import AverageMeter, save_video
from nerv.training import BaseDataModule

from vp_utils import pred_eval_step, postproc_mask, masks_to_boxes, \
    PALETTE_torch
from vp_vis import make_video, batch_draw_bbox
from models import build_model
from datasets import build_dataset

import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()


def _save_video(videos, video_fn, dim=3):
    """Save torch tensors to a video."""
    video = torch.cat(videos, dim=dim)  # [T, 3, 2*H, B*W]
    video = (video * 255.).numpy().astype(np.uint8)
    save_path = os.path.join('vis',
                             params.dataset.split('_')[0], args.params,
                             video_fn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_video(video, save_path, fps=4)


def adjust_params(params, batch_size):
    """Adjust config files for testing."""
    if batch_size > 0:
        params.val_batch_size = batch_size
    else:
        params.val_batch_size = 12 if 'obj3d' in params.dataset.lower() else 8

    # rollout the model until 50 steps for OBJ3D dataset
    if 'obj3d' in params.dataset.lower():
        num_frames = 50
    # rollout the model until 48 steps for CLEVRER dataset
    elif 'clevrer' in params.dataset.lower():
        num_frames = 48
        params.load_mask = True  # test mask/bbox
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(params.dataset))

    params.n_sample_frames = num_frames
    if params.model == 'SlotFormer':
        params.loss_dict['rollout_len'] = num_frames - params.input_frames
    else:
        raise NotImplementedError(f'Unknown model: {params.model}')

    # setup rollout image
    if params.model == 'SlotFormer':
        params.loss_dict['use_img_recon_loss'] = True
    params.load_img = True

    return params


def get_input(params, data_dict):
    """Prepare burn-in frames/gt data."""
    history_len = params.input_frames
    rollout_len = params.n_sample_frames - history_len
    gt = data_dict['img'][:, history_len:]

    if 'mask' in data_dict:
        gt_mask = data_dict['mask'][:, history_len:].long()
    else:
        gt_mask = None
    if 'bbox' in data_dict:
        gt_bbox = data_dict['bbox'][:, history_len:]
        gt_pres_mask = data_dict['pres_mask'][:, history_len:].bool()
    else:
        gt_bbox, gt_pres_mask = None, None

    assert gt.shape[1] == rollout_len

    return gt, gt_mask, gt_bbox, gt_pres_mask


def get_output(params, out_dict):
    """Extract outputs for evaluation."""
    history_len = params.input_frames
    rollout_len = params.n_sample_frames - history_len

    if params.model == 'SlotFormer':
        pred = out_dict['recon_combined']
        pred_mask = postproc_mask(out_dict['masks'])
        pred_bbox = masks_to_boxes(pred_mask, params.slot_dict['num_slots'])
    else:
        raise NotImplementedError(f'Unknown model: {params.model}')

    assert pred.shape[1] == rollout_len
    if pred_mask is not None:
        assert pred_mask.shape[1] == rollout_len
    if pred_bbox is not None:
        assert pred_bbox.shape[1] == rollout_len

    return pred, pred_mask, pred_bbox


@torch.no_grad()
def main(params):
    params = adjust_params(params, args.batch_size)

    val_set = build_dataset(params, val_only=True)
    datamodule = BaseDataModule(
        params, train_set=val_set, val_set=val_set, use_ddp=False)
    val_loader = datamodule.val_loader

    model = build_model(params).eval().cuda()
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])

    history_len = params.input_frames
    rollout_len = params.n_sample_frames - history_len
    metrics = [
        'mse', 'psnr', 'ssim', 'percept_dist', 'ari', 'fari', 'miou', 'ar'
    ]
    metric_avg_dict = {
        m: [AverageMeter() for _ in range(rollout_len)]  # per-step results
        for m in metrics
    }
    # if `args.save_num` is specified, we will save those videos and return
    # otherwise evaluate and save 10 videos
    save_videos, save_mask_videos, save_bbox_videos = [], [], []
    video_num = 10 if args.save_num <= 0 else args.save_num
    only_vis = (args.save_num > 0)
    mask_palette = PALETTE_torch.float().cuda()

    for data_dict in tqdm(val_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        gt, gt_mask, gt_bbox, gt_pres_mask = get_input(params, data_dict)
        B = gt.shape[0]

        # take model output
        out_dict = model(data_dict)
        pred, pred_mask, pred_bbox = get_output(params, out_dict)

        # compute metrics
        metric_dict = pred_eval_step(
            gt=gt,
            pred=pred,
            lpips_fn=loss_fn_vgg,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            gt_pres_mask=gt_pres_mask,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            # OBJ3D doesn't have object-level annotations
            eval_traj='clevrer' in params.dataset.lower(),
        )
        for i in range(rollout_len):
            for m in metrics:
                metric_avg_dict[m][i].update(metric_dict[m][i], B)

        # save videos for visualization
        flag = False
        for i in range(B):
            if len(save_videos) >= video_num:
                flag = only_vis
                break
            gt_video = data_dict['img'][i]  # [T, C, H, W]
            pred_video = torch.cat([gt_video[:history_len], pred[i]], dim=0)
            video = make_video(gt_video, pred_video, history_len)
            save_videos.append(video)
            # videos of per-slot seg_masks
            if gt_mask is None or pred_mask is None:
                continue
            gt_mask_video = data_dict['mask'][i].long()  # [T, H, W]
            pred_mask_video = torch.cat(
                [gt_mask_video[:history_len], pred_mask[i]], dim=0)
            gt_mask_video = mask_palette[gt_mask_video].permute(0, 3, 1, 2)
            pred_mask_video = mask_palette[pred_mask_video].permute(0, 3, 1, 2)
            mask_video = make_video(gt_mask_video, pred_mask_video,
                                    history_len)
            save_mask_videos.append(mask_video)
            # videos of image + bbox
            gt_bbox_video = data_dict['bbox'][i]  # [T, N, 4]
            pred_bbox_video = torch.cat(
                [gt_bbox_video[:history_len], pred_bbox[i]], dim=0)
            gt_bbox_video = batch_draw_bbox(gt_video, gt_bbox_video,
                                            data_dict['pres_mask'][i])
            pred_bbox_video = batch_draw_bbox(pred_video, pred_bbox_video)
            bbox_video = make_video(gt_bbox_video, pred_bbox_video,
                                    history_len)
            save_bbox_videos.append(bbox_video)

        torch.cuda.empty_cache()
        # only do some visualizations
        if flag:
            break

    if len(save_videos):
        _save_video(save_videos, f'{args.params}.mp4')
        if len(save_mask_videos):
            _save_video(save_mask_videos, f'{args.params}_mask.mp4')
            _save_video(save_bbox_videos, f'{args.params}_bbox.mp4')
    if only_vis:
        return

    # save metrics
    metrics = {
        k: np.array([m.avg for m in v])
        for k, v in metric_avg_dict.items()
    }
    save_dir = os.path.join('vis', params.dataset.split('_')[0], args.params)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in metrics.items():
        np.save(os.path.join(save_dir, f'{k}.npy'), v)
        print(f'{k}: {v.mean():.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate video prediction')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='load weight')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--save_num', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    args.params = os.path.basename(args.params)
    params = importlib.import_module(args.params)
    params = params.SlotFormerParams()
    params.ddp = False

    torch.backends.cudnn.benchmark = True
    main(params)
