import os
import sys
import pdb
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch

from nerv.utils import load_obj, dump_obj, mkdir_or_exist

from models import build_model

OBS_FRAMES = 128
TARGET_LEN = 160


@torch.no_grad()
def rollout_video_slots(model, pre_slots):
    """Returns slots including obs_slots and rollout_slots."""
    model.eval()
    torch.cuda.empty_cache()
    bs = torch.cuda.device_count()
    history_len = params.input_frames
    frame_offset = params.frame_offset
    all_fn = list(pre_slots.keys())
    all_slots = {}
    for start_idx in tqdm(range(0, len(all_fn), bs)):
        end_idx = min(start_idx + bs, len(all_fn))
        slots = [pre_slots[fn]
                 for fn in all_fn[start_idx:end_idx]]  # a list of [128, N, C]
        # to [B, 128, N, C]
        ori_slots = torch.from_numpy(np.stack(slots, axis=0))
        # pad to target len (160)
        pad_slots = torch.zeros(
            (ori_slots.shape[0], TARGET_LEN - OBS_FRAMES, ori_slots.shape[2],
             ori_slots.shape[3])).type_as(ori_slots)
        ori_slots = torch.cat((ori_slots, pad_slots), dim=1)
        ori_slots = ori_slots.float().cuda()
        obs_slots = ori_slots[:, :OBS_FRAMES]  # [B, 128, N, C]
        # for models trained with frame offset, if offset is 3
        # we rollout [0, 3, 6, ...], [1, 4, 7, ...], [2, 5, 8, ...]
        # and then concat them to [0, 1, 2, 3, 4, 5, ...]
        all_pred_slots = []
        off_range = range(frame_offset)
        for off_idx in off_range:
            start = OBS_FRAMES - history_len * frame_offset + off_idx
            in_slots = ori_slots[:, start::frame_offset]  # [B, 32+6, N, C]
            model.module.rollout_len = in_slots.shape[1] - history_len
            data_dict = {'slots': in_slots}
            pred_slots = model(data_dict)['pred_slots']  # [B, 32, N, C]
            all_pred_slots.append(pred_slots)
        pred_slots = torch.stack([
            all_pred_slots[i % frame_offset][:, i // frame_offset]
            for i in range(TARGET_LEN - OBS_FRAMES)
        ], 1)  # [B, 32, N, C]
        slots = torch.cat([obs_slots, pred_slots], dim=1)  # [B, 160, N, C]
        assert slots.shape[1] == TARGET_LEN
        for i, fn in enumerate(all_fn[start_idx:end_idx]):
            all_slots[fn] = slots[i].cpu().numpy()
        torch.cuda.empty_cache()
        del slots, pred_slots, all_pred_slots
        torch.cuda.empty_cache()
    return all_slots


def process_video(model):
    """Rollout slots using SlotFormer model."""
    # all_slots = {
    #     'train': {'1': slots, '2': slots, ...},
    #     'val': {'1': slots, '2': slots, ...},
    #     'test': {'1': slots, '2': slots, ...},
    # }
    all_slots = load_obj(params.slots_root)
    train_slots, val_slots, test_slots = \
        all_slots['train'], all_slots['val'], all_slots['test']
    len_train, len_val, len_test = \
        len(train_slots), len(val_slots), len(test_slots)

    # forward through train/val/test dataset
    print(f'Processing {params.dataset} video val set...')
    val_slots = rollout_video_slots(model, val_slots)
    print(f'Processing {params.dataset} video train set...')
    train_slots = rollout_video_slots(model, train_slots)
    print(f'Processing {params.dataset} video test set...')
    test_slots = rollout_video_slots(model, test_slots)

    try:
        slots = {'train': train_slots, 'val': val_slots, 'test': test_slots}
        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'train: {len(train_slots)}/{len_train}, '
              f'val: {len(val_slots)}/{len_val}, '
              f'test: {len(test_slots)}/{len_test}')
    except:
        pdb.set_trace()

    # create soft link to weight dir
    ln_path = os.path.join(os.path.dirname(args.weight), 'rollout_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def main():
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model = torch.nn.DataParallel(model).cuda().eval()
    process_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollout CLEVRER slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='load weight')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./data/CLEVRER/rollout_slots.pkl',
        help='path to save slots',
    )
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    params.loss_dict['use_img_recon_loss'] = False

    torch.backends.cudnn.benchmark = True
    main()
