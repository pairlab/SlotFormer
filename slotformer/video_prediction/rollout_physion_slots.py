"""Script for unrolling slots on Physion dataset."""

import os
import sys
import pdb
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch

from nerv.utils import dump_obj, mkdir_or_exist

from models import build_model
from datasets import build_dataset

OBS_FRAMES = int(30 * 1.5)  # 1.5s of the 30FPS video


@torch.no_grad()
def rollout_video_slots(model, dataset):
    """Returns slots including obs_slots and rollout_slots."""
    model.eval()
    torch.cuda.empty_cache()
    bs = torch.cuda.device_count()
    history_len = params.input_frames
    frame_offset = params.frame_offset
    all_slots = []
    range_idx = range(0, len(dataset.files), bs)
    for start_idx in tqdm(range_idx):
        end_idx = min(start_idx + bs, dataset.num_videos)
        slots = [
            dataset.video_slots[os.path.basename(fn)][:dataset.video_len]
            for fn in dataset.files[start_idx:end_idx]
        ]  # a list of [150, N, C]
        # to [B, 150, N, C]
        ori_slots = torch.from_numpy(np.stack(slots, axis=0)).float().cuda()
        obs_slots = ori_slots[:, :OBS_FRAMES]  # [B, 45, N, C]
        # for models trained with frame offset, if offset is 3
        # we rollout [0, 3, 6, ...], [1, 4, 7, ...], [2, 5, 8, ...]
        # and then concat them to [0, 1, 2, 3, 4, 5, ...]
        all_pred_slots = []
        off_range = range(frame_offset)
        for off_idx in off_range:
            start = OBS_FRAMES - history_len * frame_offset + off_idx
            in_slots = ori_slots[:, start::frame_offset]  # [B, 105+6, N, C]
            model.module.rollout_len = in_slots.shape[1] - history_len
            data_dict = {'slots': in_slots}
            pred_slots = model(data_dict)['pred_slots']  # [B, 105, N, C]
            all_pred_slots.append(pred_slots)
        pred_slots = torch.stack([
            all_pred_slots[i % frame_offset][:, i // frame_offset]
            for i in range(dataset.video_len - OBS_FRAMES)
        ], 1)  # [B, 105, N, C]
        slots = torch.cat([obs_slots, pred_slots], dim=1)  # [B, 150, N, C]
        assert slots.shape[1] == dataset.video_len
        all_slots += [slot for slot in slots.detach().cpu().numpy()]
        torch.cuda.empty_cache()
        del slots, pred_slots, all_pred_slots
        torch.cuda.empty_cache()
    all_slots = np.stack(all_slots, axis=0)  # [N, T, n, c]
    return all_slots


def process_video(model):
    """Rollout slots using SlotFormer model."""
    train_set, val_set = build_dataset(params)

    # forward through train/val dataset
    print(f'Processing {params.dataset} video val set...')
    val_slots = rollout_video_slots(model, val_set)
    print(f'Processing {params.dataset} video train set...')
    train_slots = rollout_video_slots(model, train_set)

    try:
        train_slots = {
            os.path.basename(train_set.files[i]): train_slots[i]
            for i in range(len(train_slots))
        }  # each embedding is of shape [T, N, C]
        val_slots = {
            os.path.basename(val_set.files[i]): val_slots[i]
            for i in range(len(val_slots))
        }
        slots = {'train': train_slots, 'val': val_slots}
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'train: {len(train_slots)}/{train_set.num_videos}, '
              f'val: {len(val_slots)}/{val_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to weight dir
    ln_path = os.path.join(os.path.dirname(args.weight), 'readout_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def process_test_video(model):
    """Rollout slots using SlotFormer model."""
    test_set = build_dataset(params)

    # forward through test dataset
    print(f'Processing {params.dataset} video test set...')
    test_slots = rollout_video_slots(model, test_set)

    try:
        test_slots = {
            os.path.basename(test_set.files[i]): test_slots[i]
            for i in range(len(test_slots))
        }  # each embedding is of shape [T, N, C]
        slots = {'test': test_slots}
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'test: {len(test_slots)}/{test_set.num_videos}')
    except:
        pdb.set_trace()

    # create soft link to weight dir
    ln_path = os.path.join(os.path.dirname(args.weight), 'test_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def main():
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model = torch.nn.DataParallel(model).cuda().eval()
    if 'test' in params.dataset:
        process_test_video(model)
    else:
        process_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollout Physion slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--subset', type=str, default='readout')
    parser.add_argument(
        '--weight', type=str, required=True, help='load weight')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./data/Physion/rollout_readout_slots.pkl',
        help='path to save slots',
    )
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    assert 'physion' in args.params and args.subset in ['readout', 'test'], \
        'should only be used to perform readout/testing on Physion dataset'
    assert args.subset in args.save_path, \
        'please include `subset` in `save_path` to differentiate slot files'
    # switch to $SUBSET slots
    params.dataset = f'physion_slots_{args.subset}'
    slot_name = f'{args.subset}_slots.pkl'
    params.slots_root = os.path.join(
        os.path.dirname(params.slots_root), slot_name)
    params.loss_dict['use_img_recon_loss'] = False

    torch.backends.cudnn.benchmark = True
    main()
