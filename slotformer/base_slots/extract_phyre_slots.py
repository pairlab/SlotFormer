import os
import sys
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset


@torch.no_grad()
def extract_phyre_video_slots(model, dataset):
    """Returns slots extracted from each video of the dataset."""
    model.eval()
    slot_key = 'post_slots' if params.model == 'StoSAVi' else 'slots'
    save_root = os.path.join(
        args.save_path,
        'slots',
        os.path.basename(args.params),
        f'{dataset.protocal}-fold_{str(dataset.fold)}-{dataset.split}-'
        f'data_{str(dataset.ratio)}-pos_{str(dataset.pos_ratio)}',
    )
    os.makedirs(save_root, exist_ok=True)
    torch.cuda.empty_cache()

    # create soft link to the weight dir
    if args.split in [-1, 0]:
        ln_path = os.path.join(
            os.path.dirname(args.weight), f'{str(dataset.split)}_slots')
        os.system(r'ln -s {} {}'.format(save_root, ln_path))

    # load subset of videos from the dataset
    dataset.vid_len = args.vid_len * dataset.fps
    dataset.load_video = True
    bs = args.bs
    total_num = len(dataset)
    if args.split != -1:
        start_idx = total_num // args.total_split * args.split
        end_idx = total_num // args.total_split * (args.split + 1) if \
            args.split < (args.total_split - 1) else total_num
        # check how many files have already been processed
        print('Filtering already processed videos...')
        for idx in tqdm(range(start_idx, end_idx)):
            npy_path = os.path.join(save_root, f'{idx:06d}.npy')
            if (not os.path.exists(npy_path)):
                break
        start_idx = max(idx - 1, 0)  # in case the last file is corrupted
        dataset.start_idx = start_idx
        dataset.end_idx = end_idx

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=args.cpus,
        pin_memory=True,
        drop_last=False,
    )

    for data_dict in tqdm(dataloader):
        in_dict = {'img': data_dict['video'].float().cuda()}
        out_dict = model(in_dict)
        slots = out_dict[slot_key].detach().cpu().numpy()  # [B, T, n, c]
        torch.cuda.empty_cache()
        # save slots to individual npy files
        idx = data_dict['data_idx'].numpy()
        for i, save_idx in enumerate(idx):
            vid_len = data_dict['vid_len'][i].item()
            np.save(
                os.path.join(save_root, f'{save_idx:06d}.npy'),
                slots[i, :vid_len],  # only save to the real video length
            )


def process_video(model):
    """Extract slot_embs using video SlotAttn model"""
    train_set, val_set = build_dataset(params)

    # forward through train/val dataset
    print(f'Processing {params.dataset} video val set...')
    extract_phyre_video_slots(model, val_set)
    print(f'Processing {params.dataset} video train set...')
    extract_phyre_video_slots(model, train_set)


def main():
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model.testing = True  # we just want slots
    model = model.cuda().eval()
    process_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract PHYRE slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='pretrained model weight')
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,  # './data/PHYRE'
        help='path to save slots',
    )
    parser.add_argument('--vid_len', type=int, default=11)
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--total_split', type=int, default=10)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--cpus', type=int, default=8)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()

    assert torch.cuda.device_count() == 1
    torch.backends.cudnn.benchmark = True
    main()
