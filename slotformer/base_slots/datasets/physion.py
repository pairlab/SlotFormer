import os
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

from nerv.utils import load_obj, read_all_lines

from .utils import BaseTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PhysionDataset(Dataset):
    """Dataset for loading Physion videos."""

    def __init__(
        self,
        data_root,
        split,
        tasks,
        physion_transform,
        n_sample_frames=6,
        frame_offset=None,
        video_len=150,
        subset='training',
    ):

        if subset in ['training', 'readout']:
            assert split in ['train', 'val']
        elif subset == 'test':
            assert split == 'test'
        else:
            raise NotImplementedError(f'Unknown subset: {subset}')

        self.data_root = data_root
        self.split = split
        self.tasks = tasks
        self.physion_transform = physion_transform
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.video_len = video_len
        self.subset = subset

        # Get all numbers
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _read_frames(self, idx):
        folder, start_idx = self._get_video_start_idx(idx)
        # read frames saved from videos
        assert osp.exists(folder), "Please extract frames from videos first."
        filename = osp.join(folder, '{:06d}.jpg')
        frames = [
            Image.open(filename.format(start_idx +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(self.n_sample_frames)
        ]
        # raise error if any frame is corrupted
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [self.physion_transform(img) for img in frames]
        return torch.stack(frames, dim=0)  # [N, C, H, W]

    def _read_tokens(self, idx):
        """Load pre-computed dVAE tokens for training STEVE."""
        folder, start_idx = self._get_video_start_idx(idx)
        npy_file = folder.replace('TrainMP4s/', f'TrainNpys-{self.dvae_path}/').\
            replace('TestMP4s/', f'TestNpys-{self.dvae_path}/') + '.npy'
        if not osp.exists(npy_file):
            return None
        tokens = np.load(npy_file)  # [T, h*w]
        tokens = [
            tokens[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(tokens, axis=0).astype(np.int32)  # [N, h*w]

    def get_video(self, video_idx):
        folder = self.files[video_idx]
        # read frames saved from videos
        assert osp.exists(folder), "Please extract frames from videos first."
        num_frames = self.video_len // self.frame_offset
        frames = [
            Image.open(osp.join(
                folder, f'{n * self.frame_offset:06d}.jpg')).convert('RGB')
            for n in range(num_frames)
        ]
        # corrupted video
        if any(frame is None for frame in frames):
            return self._rand_another(is_video=True)
        frames = [self.physion_transform(img) for img in frames]
        return {
            'video': torch.stack(frames, dim=0),
            'data_idx': video_idx,
        }

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - token_id: [T, h*w], pre-computed dVAE tokens for training STEVE
        """
        if self.load_video:
            return self.get_video(idx)

        try:
            frames = self._read_frames(idx)
            tokens = self._read_tokens(idx)
        except ValueError:
            return self._rand_another()
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        if tokens is not None:
            data_dict['token_id'] = tokens
        return data_dict

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        # get the train-val split file for each subset
        # by default we store it under './splits/Physion'
        cur_dir = osp.dirname(osp.realpath(__file__))
        json_fn = osp.join(cur_dir,
                           f'splits/Physion/{self.subset}_{self.split}.json')
        json_file = load_obj(json_fn)
        # count the number of each task
        # ['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support']
        self.all_tasks = sorted(list(json_file.keys()))
        self.task2num = {task: len(json_file[task]) for task in self.all_tasks}
        self.video_idx2task_idx = {}  # mapping from video index to task index
        # load video paths
        self.files = []
        if self.tasks[0].lower() == 'all':
            print('Loading all tasks from Physion...')
            self.tasks = list(json_file.keys())
        for task in self.tasks:
            idx1 = len(self.files)
            task_files = json_file[task]  # 'xxx.mp4'
            task_files = [osp.join(self.data_root, f[:-4]) for f in task_files]
            self.files.extend(task_files)
            idx2 = len(self.files)
            self.video_idx2task_idx.update(
                {idx: self.all_tasks.index(task)
                 for idx in range(idx1, idx2)})
        self.num_videos = len(self.files)
        for folder in self.files:
            # simply use random uniform sampling
            if self.split == 'train':
                max_start_idx = self.video_len - \
                    (self.n_sample_frames - 1) * self.frame_offset
                valid_idx += [(folder, idx) for idx in range(max_start_idx)]
            # only test once per video
            else:
                size = self.n_sample_frames * self.frame_offset
                start_idx = []
                for idx in range(0, self.video_len - size + 1, size):
                    start_idx += [i + idx for i in range(self.frame_offset)]
                valid_idx += [(folder, idx) for idx in start_idx]
        return valid_idx

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)


class PhysionSlotsDataset(PhysionDataset):
    """Dataset for loading Physion videos and pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        split,
        tasks,
        physion_transform,
        n_sample_frames=25,
        frame_offset=None,
        video_len=150,
        subset='training',
        load_img=False,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            tasks=tasks,
            physion_transform=physion_transform,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            video_len=video_len,
            subset=subset,
        )

        # pre-computed slots
        self.video_slots = video_slots
        self.load_img = load_img

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _read_slots(self, idx):
        """Read video frames slots."""
        folder, start_idx = self._get_video_start_idx(idx)
        slots = self.video_slots[os.path.basename(folder)]  # [T, N, C]
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - slots: [T, N, C] slots extracted from OBJ3D video frames
            - token_id: [T, h*w], pre-computed dVAE tokens for training STEVE
        """
        try:
            slots = self._read_slots(idx)
            data_dict = {'slots': slots}
            if self.load_img:
                frames = self._read_frames(idx)
                tokens = self._read_tokens(idx)
                data_dict['img'] = frames
                if tokens is not None:
                    data_dict['token_id'] = tokens
        except ValueError:
            return self._rand_another()
        data_dict['data_idx'] = idx
        return data_dict


class PhysionSlotsLabelDataset(PhysionSlotsDataset):
    """Dataset for loading Physion videos, slots and VQA labels."""

    def __init__(
        self,
        data_root,
        video_slots,
        split,
        tasks,
        physion_transform,
        n_sample_frames=15,
        frame_offset=None,
        video_len=150,
        subset='training',
        load_img=False,
    ):

        if subset == 'readout':
            label_fn = 'PhysionTrainMP4s/readout_labels.csv'
        elif subset == 'test':
            label_fn = 'PhysionTestMP4s/labels.csv'
        else:
            raise NotImplementedError
        self.labels = pd.read_csv(os.path.join(data_root, label_fn))

        super().__init__(
            data_root=data_root,
            video_slots=video_slots,
            split=split,
            tasks=tasks,
            physion_transform=physion_transform,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            video_len=video_len,
            subset=subset,
            load_img=load_img,
        )

        # get sample index
        assert frame_offset == 1
        self.sample_idx = list(range(video_len))

        if subset == 'readout':
            return
        # filter out bad stimuli in test set
        cur_dir = osp.dirname(osp.realpath(__file__))
        bad_stimuli = read_all_lines(
            os.path.join(cur_dir, 'splits/Physion/bad_stimuli.txt'))
        remove_files = []
        for file in self.files:
            file_to_check = file.replace('-redyellow', '')
            if any(s in file_to_check for s in bad_stimuli):
                remove_files.append(file)
        self.files = [f for f in self.files if f not in remove_files]
        print(f'remove {len(remove_files)} files, now have {len(self.files)}')

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _read_frames(self, file_idx):
        """Read the entire video."""
        folder = self.files[file_idx]
        frames = [
            Image.open(osp.join(folder, f'{i:06d}.jpg')).convert('RGB')
            for i in self.sample_idx
        ]
        # raise error if any frame is corrupted
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [self.physion_transform(img) for img in frames]
        return torch.stack(frames, dim=0).float()

    def _read_slots(self, file_idx):
        """Read slots of the entire video."""
        folder = self.files[file_idx]
        slots = self.video_slots[os.path.basename(folder)]  # [T, N, C]
        slots = [slots[i] for i in self.sample_idx]
        return np.stack(slots, axis=0).astype(np.float32)

    def _read_label(self, file_idx):
        """Load the VQA label."""
        csv_key = os.path.basename(self.files[file_idx])  # 'xxx_img.mp4'
        if csv_key.endswith('.mp4'):
            csv_key = csv_key[:-4]
        if self.subset == 'readout' and csv_key.endswith('_img'):
            csv_key = csv_key[:-4]
        if self.subset == 'test' and '-redyellow' in csv_key:
            csv_key = csv_key.replace('-redyellow', '')
        label = self.labels[self.labels['Unnamed: 0'] ==
                            csv_key]['ground truth outcome'].item()
        label = 1 if label else 0
        return label

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - task_idx: int
            - img: [T, 3, H, W]
            - slots: [T, N, C] slots extracted from OBJ3D video frames
            - label: 1: success, 0: fail
        """
        file_idx = idx
        try:
            slots = self._read_slots(file_idx)
            label = self._read_label(file_idx)
            data_dict = {'slots': slots, 'label': label}
            if self.load_img:
                frames = self._read_frames(file_idx)
                data_dict['img'] = frames
        except ValueError:
            return self._rand_another()
        data_dict['data_idx'] = idx
        data_dict['task_idx'] = self.video_idx2task_idx[file_idx]
        return data_dict

    def __len__(self):
        # one data is a (video, label) pair
        return len(self.files)


def build_physion_dataset(params, val_only=False):
    """Build Physion video dataset."""
    subset = params.dataset.split('_')[-1]
    physion_transform = BaseTransforms(params.resolution)
    args = dict(
        data_root=params.data_root,
        split='val',
        tasks=params.tasks,
        physion_transform=physion_transform,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        subset=subset,
    )
    if subset == 'test':
        args['split'] = 'test'
        val_only = True
    val_dataset = PhysionDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = PhysionDataset(**args)
    # an ugly hack to find the path of pre-computed dVAE token_ids
    dvae_path = params.dvae_dict['dvae_ckp_path'].split('/')[1] if \
        hasattr(params, 'dvae_dict') else 'dvae-none'
    assert 'dvae' in dvae_path
    train_dataset.dvae_path = dvae_path
    val_dataset.dvae_path = dvae_path
    return train_dataset, val_dataset


def build_physion_slots_dataset(params, val_only=False):
    """Build Physion video dataset with slots."""
    subset = params.dataset.split('_')[-1]
    physion_transform = BaseTransforms(params.resolution)
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=None,
        split='val',
        tasks=params.tasks,
        physion_transform=physion_transform,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        subset=subset,
        load_img=params.loss_dict['use_img_recon_loss'],
    )
    if subset == 'test':
        args['split'] = 'test'
        args['video_slots'] = slots['test']
        val_only = True
    else:
        args['video_slots'] = slots['val']
    val_dataset = PhysionSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = PhysionSlotsDataset(**args)
    # an ugly hack to find the path of pre-computed dVAE token_ids
    dvae_path = params.dvae_dict['dvae_ckp_path'].split('/')[1] if \
        hasattr(params, 'dvae_dict') else 'dvae-none'
    assert 'dvae' in dvae_path
    train_dataset.dvae_path = dvae_path
    val_dataset.dvae_path = dvae_path
    return train_dataset, val_dataset


def build_physion_slots_label_dataset(params, val_only=False):
    """Build Physion video dataset with slots and labels."""
    subset = params.dataset.split('_')[-1]
    physion_transform = BaseTransforms(params.resolution)
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=None,
        split='val',
        tasks=params.tasks,
        physion_transform=physion_transform,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        subset=subset,
    )
    if subset == 'test':
        args['split'] = 'test'
        args['video_slots'] = slots['test']
        val_only = True
    else:
        args['video_slots'] = slots['val']
    val_dataset = PhysionSlotsLabelDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = PhysionSlotsLabelDataset(**args)
    return train_dataset, val_dataset
