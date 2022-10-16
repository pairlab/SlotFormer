import os
import sys
import copy
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import phyre
import torch
from torch.utils.data import Dataset, DataLoader

from models import build_model
from slotformer.base_slots.datasets.phyre import fix_video_len


def make_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.cpus,
        pin_memory=True,
        drop_last=False,
    )


class PHYREDataset(Dataset):
    """Loading the first slot for rollout."""

    def __init__(
        self,
        data_root,
        slot_root,
        split,
        protocal='within',
        fold=0,
        vid_len=15,
        ratio=1.,
        pos_ratio=0.2,
        start_idx=None,
        end_idx=None,
    ):

        self.data_root = data_root
        self.slot_root = slot_root
        self.split = split

        self.protocal = protocal
        self.fold = fold
        self.vid_len = vid_len
        self.ratio = ratio
        self.pos_ratio = pos_ratio

        self.start_idx = start_idx
        self.end_idx = end_idx

        self._filter_actions()

    def _read_slots(self, idx):
        slot_path = os.path.join(self.slot_root, f'{idx:06d}.npy')
        slots = np.load(slot_path)
        slots = fix_video_len(slots, self.vid_len)
        return slots

    def _rand_another(self, idx):
        """Random get another sample when encountering loading error."""
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def __getitem__(self, idx):
        idx = idx if self.start_idx is None else idx + self.start_idx
        data_dict = {'data_idx': idx, 'error_flag': False}
        try:
            slots = self._read_slots(idx)
            data_dict['slots'] = slots
        except FileNotFoundError:
            data_dict = self._rand_another(idx)
            data_dict['error_flag'] = True
        return data_dict

    def __len__(self):
        if self.start_idx is None:
            return self.video_info.shape[0]
        return self.end_idx - self.start_idx

    def _filter_actions(self):
        """Filter actions that generate videos longer than a length."""
        split = self.split
        protocal = self.protocal  # 'within' or 'cross'
        fold = self.fold
        ratio = self.ratio  # data_ratio
        pos_ratio = self.pos_ratio  # ratio of positive actions

        eval_setup = f'ball_{protocal}_template'
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold)
        tasks = train_tasks + dev_tasks if split == 'train' else test_tasks
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # filter tasks
        candidate_lst = [f'{i:05d}' for i in range(0, 25)]
        tasks = [task for task in tasks if task.split(':')[0] in candidate_lst]
        self.simulator = phyre.initialize_simulator(tasks, action_tier)

        # load pre-generated data
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(
            cur_dir,
            '../base_slots/datasets/splits/PHYRE',
            f'{protocal}-fold_{str(fold)}-{split}-data_{str(ratio)}-'
            f'pos_{str(pos_ratio)}.npy',
        )
        label_path = info_path.replace('.npy', '-label.npy')
        assert os.path.exists(info_path) and os.path.exists(label_path)
        self.video_info = np.load(info_path)
        self.act_labels = np.load(label_path)


@torch.no_grad()
def rollout_video_slots(model, dataset):
    """Returns slots including obs_slots and rollout_slots."""
    dataloader = make_dataloader(dataset)
    save_root = os.path.join(
        args.save_path,
        'rollout_slots',
        os.path.basename(args.params),
        f'{dataset.protocal}-fold_{str(dataset.fold)}-{dataset.split}',
    )
    os.makedirs(save_root, exist_ok=True)

    # create soft link to the weight dir
    if args.split in [-1, 0]:
        ln_path = os.path.join(
            os.path.dirname(args.weight), f'{str(dataset.split)}_slots')
        os.system(r'ln -s {} {}'.format(save_root, ln_path))

    for batch_data in tqdm(dataloader):
        data_idx = batch_data.pop('data_idx').numpy()
        error_flag = batch_data.pop('error_flag').numpy()
        # check if already extracted
        if all(
                os.path.exists(os.path.join(save_root, f'{i:06d}.npy'))
                for i in data_idx):
            continue
        in_dict = {'slots': batch_data['slots'].float().cuda()}
        out_dict = model(in_dict)
        slots = out_dict['pred_slots'].cpu()  # [B, T, N, C]
        slots = torch.cat([batch_data['slots'][:, :1], slots], dim=1)
        assert slots.shape[1] == batch_data['slots'].shape[1]
        torch.cuda.empty_cache()
        # save slots to individual npy files
        for i in range(slots.shape[0]):
            if error_flag[i]:
                continue
            np.save(
                os.path.join(save_root, f'{data_idx[i]:06d}.npy'), slots[i])


def process_video(model):
    """Rollout slots using SlotFormer model."""
    # build dataset
    val_args = dict(
        data_root=params.data_root,
        slot_root=params.slots_root.format('val'),
        split='val',
        protocal=params.phyre_protocal,
        fold=params.phyre_fold,
        vid_len=params.video_len,
        ratio=params.data_ratio,
        pos_ratio=params.pos_ratio,
    )
    val_set = PHYREDataset(**val_args)
    train_args = copy.deepcopy(val_args)
    train_args['split'] = 'train'
    train_args['slot_root'] = params.slots_root.format('train')
    train_set = PHYREDataset(**train_args)
    # make splits
    num_train, num_val = len(train_set), len(val_set)
    if args.split != -1:
        train_start_idx = num_train // args.total_split * args.split
        train_end_idx = num_train // args.total_split * (args.split + 1) if \
            args.split < (args.total_split - 1) else num_train
        val_start_idx = num_val // args.total_split * args.split
        val_end_idx = num_val // args.total_split * (args.split + 1) if \
            args.split < (args.total_split - 1) else num_val
        train_set.start_idx = train_start_idx
        train_set.end_idx = train_end_idx
        val_set.start_idx = val_start_idx
        val_set.end_idx = val_end_idx

    print(f'Processing {params.dataset} video val set...')
    rollout_video_slots(model, val_set)
    print(f'Processing {params.dataset} video train set...')
    rollout_video_slots(model, train_set)


def main():
    assert torch.cuda.device_count() == 1, 'only support single GPU'
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model = model.eval().cuda()
    process_video(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollout PHYRE slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument(
        '--weight', type=str, required=True, help='load weight')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./data/PHYRE',
        help='path to save slots',
    )
    parser.add_argument('--vid_len', type=int, default=-1)
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--total_split', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--cpus', type=int, default=8)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    # adjust rollout length
    if args.vid_len > 0:
        params.video_len = args.vid_len * params.fps
    params.n_sample_frames = params.video_len
    params.loss_dict['rollout_len'] = params.video_len - 1
    params.loss_dict['use_img_recon_loss'] = False

    torch.backends.cudnn.benchmark = True
    main()
