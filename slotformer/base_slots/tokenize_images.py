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
def extract_video_tokens(model, dataset):
    """Returns tokens extracted from each video of the dataset."""
    model.eval()
    torch.cuda.empty_cache()
    all_tokens = []

    dataset.load_video = True
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs * torch.cuda.device_count(),
        shuffle=False,
        num_workers=args.cpus * torch.cuda.device_count(),
        pin_memory=True,
        drop_last=False,
    )

    for data_dict in tqdm(dataloader):
        data_dict = {'img': data_dict['video'].float().cuda()}
        tokens = model(data_dict)
        # [B, T, h, w] --> [B, T, h*w]
        tokens = tokens.flatten(2, 3).detach().cpu().numpy().astype(np.int16)
        all_tokens += [token for token in tokens]
        torch.cuda.empty_cache()
    all_tokens = np.stack(all_tokens, axis=0).astype(np.int16)  # [N, T, h*w]
    return all_tokens


def process_physion_training_video(model):
    """Tokenize Physion videos to tokens"""
    dvae_path = os.path.basename(args.params)
    print(f'Tokenizing Physion training videos using {dvae_path}...')
    train_set, val_set = build_dataset(params)

    # forward through train/val dataset
    print('Processing PhysionTrain video val set...')
    val_tokens = extract_video_tokens(model, val_set)
    print('Processing PhysionTrain video train set...')
    train_tokens = extract_video_tokens(model, train_set)

    # save to individual npy files
    # 'Physion/PhysionTrainMP4s/xxx/1.mp4' goes to
    # 'Physion/PhysionTrainMP4s-$dvae_path/xxx/1.npy'
    cnt = 0
    for i in range(len(train_tokens)):
        mp4_fn = train_set.files[i]
        npy_fn = mp4_fn.replace('TrainMP4s/', f'TrainNpys-{dvae_path}/').\
            replace('TestMP4s/', f'TestNpys-{dvae_path}/').replace('.mp4', '.npy')
        os.makedirs(os.path.dirname(npy_fn), exist_ok=True)
        np.save(npy_fn, train_tokens[i])  # each token is of shape [T, h*w]
        cnt += 1
    print(f'train: {cnt}/{train_set.num_videos}')
    cnt = 0
    for i in range(len(val_tokens)):
        mp4_fn = val_set.files[i]
        npy_fn = mp4_fn.replace('TrainMP4s/', f'TrainNpys-{dvae_path}/').\
            replace('TestMP4s/', f'TestNpys-{dvae_path}/').replace('.mp4', '.npy')
        os.makedirs(os.path.dirname(npy_fn), exist_ok=True)
        np.save(npy_fn, val_tokens[i])  # each token is of shape [T, h*w]
        cnt += 1
    print(f'val: {cnt}/{val_set.num_videos}')


def main():
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model.testing = True  # we just want tokens
    model = torch.nn.DataParallel(model).cuda().eval()
    process_physion_training_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize videos')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='pretrained dvae weight')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--cpus', type=int, default=8)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()

    torch.backends.cudnn.benchmark = True
    main()
