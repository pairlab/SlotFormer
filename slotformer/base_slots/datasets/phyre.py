import os
import numpy as np
from tqdm import tqdm

import phyre

import torch
from torch.utils.data import Dataset

from .utils import BaseTransforms


def _hex_to_ints(hex_string):
    hex_string = hex_string.strip('#')
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


WAD_COLORS = np.array(
    [
        [255, 255, 255],  # White.
        _hex_to_ints('f34f46'),  # Red.
        _hex_to_ints('6bcebb'),  # Green.
        _hex_to_ints('1877f2'),  # Blue.
        _hex_to_ints('4b4aa4'),  # Purple.
        _hex_to_ints('b9cad2'),  # Gray.
        [0, 0, 0],  # Black.
        _hex_to_ints('fcdfe3'),  # Light red.
    ],
    dtype=np.uint8)

# reverse white and black
WAD_REVERSE_COLOR = np.array(
    [
        [0, 0, 0],  # Black.
        _hex_to_ints('f34f46'),  # Red.
        _hex_to_ints('6bcebb'),  # Green.
        _hex_to_ints('1877f2'),  # Blue.
        _hex_to_ints('4b4aa4'),  # Purple.
        _hex_to_ints('b9cad2'),  # Gray.
        [255, 255, 255],  # White.
        _hex_to_ints('fcdfe3'),  # Light red.
    ],
    dtype=np.uint8)


def observations_to_uint8_rgb(scene, reverse=False):
    """Convert an observation as returned by a simulator to an image."""
    if reverse:
        base_image = WAD_REVERSE_COLOR[scene]
    else:
        base_image = WAD_COLORS[scene]
    base_image = base_image[::-1]
    return base_image


def get_last_moving_idx(images):
    """Get the last index of the frame where objects are still moving."""
    idx = np.argmax([(images[i] == images[i + 1]).all()
                     for i in range(len(images) - 1)])
    # all frames are different
    if idx == 0:
        return len(images) - 1
    # images[idx], images[idx+1], ... are the same
    return idx


def fix_video_len(video, N):
    """Dup or crop the video to a desired length N."""
    if len(video) < N:
        video = np.concatenate([video, *([video[-1:]] * (N - len(video)))])
    elif len(video) > N:
        video = video[:N]
    return video


class PHYREDataset(Dataset):
    """Dataset for loading PHYRE videos."""

    def __init__(
        self,
        data_root,
        split,
        phyre_transform,
        seq_size=6,
        frame_offset=1,
        fps=1,
        protocal='within',
        fold=0,
        vid_len=15,
        ratio=1.,
        pos_ratio=0.2,
        reverse_color=False,
    ):

        self.data_root = data_root
        self.split = split
        self.phyre_transform = phyre_transform
        self.resolution = phyre_transform.resolution

        self.seq_size = seq_size
        self.fps = fps  # simulation fps
        self.frame_offset = frame_offset
        assert self.frame_offset == 1, 'should modify fps instead'

        self.protocal = protocal  # 'within' or 'cross'
        self.fold = fold  # 0~9
        self.vid_len = vid_len
        self.ratio = ratio  # only use a portion of the whole dataset
        self.pos_ratio = pos_ratio  # balance pos and neg action samples
        self.reverse_color = reverse_color  # make background black

        self._filter_actions()
        self.files = self.video_info  # for compatibility

        # by default, we load small video clips
        self.load_video = False
        # only load a subset of all trials
        self.start_idx = None
        self.end_idx = None

    def _rand_another(self, idx, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = (idx + 10) % len(self)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def get_video(self, idx, video_len=None):
        video_len = self.vid_len if video_len is None else video_len
        task_id, acts = self.video_info[idx, 0], self.video_info[idx, 1:]
        sim = self.simulator.simulate_action(
            int(task_id),
            acts,
            stride=60 // self.fps,
            need_images=True,
            need_featurized_objects=False,
        )
        images = sim.images[::self.frame_offset]
        vid_len = min(len(images), video_len)
        images = fix_video_len(images, video_len)  # dup or crop to a fixed len
        frames = [
            self.phyre_transform(self._preproc_img(img)) for img in images
        ]
        label = int(sim.status == 1)  # 1: success, 0: fail
        assert label == self.act_labels[idx], \
            'simulated label does not match pre-generated label'
        data_dict = {
            'video': torch.stack(frames, dim=0),  # [T, C, H, W]
            'data_idx': idx,
            'label': label,
            'vid_len': vid_len,  # the real length of the video
        }
        return data_dict

    def _read_frames(self, idx, video_len=None):
        # if None we discard this sample if its length is smaller than seq_size
        # otherwise we pad it to the desired length
        pad_img = (video_len is not None)
        video_len = self.seq_size if video_len is None else video_len
        task_id, acts = self.video_info[idx, 0], self.video_info[idx, 1:]
        sim = self.simulator.simulate_action(
            int(task_id),
            acts,
            stride=60 // self.fps,
            need_images=True,
            need_featurized_objects=False,
        )
        images = sim.images[::self.frame_offset]
        vid_len = min(len(images), video_len)
        # get all non-static frames
        last_idx = get_last_moving_idx(images)
        images = images[:last_idx + 1]
        # if this trial is too short, we discard it
        # otherwise we pad it to the desired length
        if len(images) < video_len:
            if not pad_img:
                raise ValueError
            else:
                images = fix_video_len(images, video_len)
        # read from the beginning of the simulation
        start_idx = 0
        images = images[start_idx:start_idx + video_len]
        frames = [
            self.phyre_transform(self._preproc_img(img)) for img in images
        ]
        frames = torch.stack(frames, dim=0)  # [N, C, H, W]
        label = int(sim.status == 1)  # 1: success, 0: fail
        assert label == self.act_labels[idx], \
            'simulated label does not match pre-generated label'
        data_dict = {'img': frames, 'label': label, 'vid_len': vid_len}
        return data_dict

    def _preproc_img(self, img):
        img = observations_to_uint8_rgb(img, reverse=self.reverse_color)
        return np.ascontiguousarray(img)

    def __getitem__(self, idx):
        """Data dict:
            - img: [T, 3, H, W]
            - label: 1: success, 0: fail
            - vid_len: the real length of the video
        """
        # load entire video for testing
        if self.load_video:
            if self.start_idx is not None:
                idx = self.start_idx + idx
            return self.get_video(idx)
        try:
            data_dict = self._read_frames(idx)
        except ValueError:
            return self._rand_another(idx)
        data_dict['data_idx'] = idx
        return data_dict

    def __len__(self):
        if self.load_video and self.start_idx is not None:
            return self.end_idx - self.start_idx
        return self.video_info.shape[0]

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

        # load pre-generated data if possible
        # by default we store it under './splits/PHYRE'
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(
            cur_dir,
            'splits/PHYRE',
            f'{protocal}-fold_{str(fold)}-{split}-data_{str(ratio)}-'
            f'pos_{str(pos_ratio)}.npy',
        )
        label_path = info_path.replace('.npy', '-label.npy')
        if os.path.exists(info_path) and os.path.exists(label_path):
            self.video_info = np.load(info_path)
            self.act_labels = np.load(label_path)
            return

        # all the actions
        cache = phyre.get_default_100k_cache('ball')
        training_data = cache.get_sample(tasks, None)
        # (100000 x 3)
        actions = training_data['actions']
        # (num_tasks x 100000)
        sim_statuses = training_data['simulation_statuses']

        # filter actions, here we follow RPIN
        num_pos = int(2000 * pos_ratio) if \
            split == 'train' else int(500 * pos_ratio)
        num_neg = int(2000 * (1 - pos_ratio)) if \
            split == 'train' else int(500 * (1 - pos_ratio))
        num_pos = int(ratio * num_pos)
        num_neg = int(ratio * num_neg)

        # keep the same random seed
        np.random.seed(fold)

        self.video_info = np.zeros((0, 4))
        self.act_labels = np.zeros(0)
        for t_id, t in enumerate(tqdm(tasks)):
            sim_status = sim_statuses[t_id]
            pos_acts = actions[sim_status == 1].copy()
            neg_acts = actions[sim_status == -1].copy()
            np.random.shuffle(pos_acts)
            np.random.shuffle(neg_acts)
            pos_acts = pos_acts[:num_pos]
            neg_acts = neg_acts[:num_neg]
            acts = np.concatenate([pos_acts, neg_acts])

            video_info = np.zeros((acts.shape[0], 4))
            video_info[:, 0] = t_id
            video_info[:, 1:] = acts
            self.video_info = np.concatenate([self.video_info, video_info])

            act_labels = np.concatenate(
                [np.ones(len(pos_acts)),
                 np.zeros(len(neg_acts))])
            self.act_labels = np.concatenate([self.act_labels, act_labels])

        self.act_labels = self.act_labels.astype(np.int32)
        assert len(self.video_info) == len(self.act_labels)
        print('Total number of actions:', self.video_info.shape[0])
        print('Number of positive actions:', (self.act_labels == 1).sum())
        print('Number of negative actions:', (self.act_labels == 0).sum())

        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        np.save(info_path, self.video_info)
        np.save(label_path, self.act_labels)


class PHYRESlotsDataset(PHYREDataset):
    """Dataset for loading PHYRE videos and pre-computed slots."""

    def __init__(
        self,
        data_root,
        slot_root,
        split,
        phyre_transform,
        seq_size=6,
        frame_offset=1,
        fps=1,
        protocal='within',
        fold=0,
        vid_len=15,
        ratio=1.,
        pos_ratio=0.2,
        reverse_color=False,
        load_img=False,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            phyre_transform=phyre_transform,
            seq_size=seq_size,
            frame_offset=frame_offset,
            fps=fps,
            protocal=protocal,
            fold=fold,
            vid_len=vid_len,
            ratio=ratio,
            pos_ratio=pos_ratio,
            reverse_color=reverse_color,
        )

        self.slot_root = slot_root
        self.load_img = load_img

    def _rand_another(self, idx, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = (idx + 10) % len(self)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _read_slots(self, idx, video_len=None):
        video_len = self.seq_size if video_len is None else video_len
        slots_path = os.path.join(self.slot_root, f'{idx:06d}.npy')
        slots = np.load(slots_path).astype(np.float32)[::self.frame_offset]
        slots = np.ascontiguousarray(slots)
        vid_len = min(len(slots), video_len)
        slots = fix_video_len(slots, video_len)
        label = self.act_labels[idx]
        return {'slots': slots, 'vid_len': vid_len, 'label': label}

    def __getitem__(self, idx):
        """Data dict:
            - img: [T, 3, H, W]
            - slots: [T, N, C]
            - label: 1: success, 0: fail
            - vid_len: the real length of the video
        """
        # we cannot directly use super().__getitem__()
        # because if `self._read_frames()` raises en error
        # it will rand another index to load action/frames
        # however, we don't know that idx, so the loaded slots will mismatch
        try:
            data_dict = self._read_slots(idx)
            if self.load_img:
                img_dict = self._read_frames(idx, video_len=self.seq_size)
                assert len(data_dict['slots']) == len(img_dict['img'])
                data_dict['img'] = img_dict['img']
                data_dict['vid_len'] = min(data_dict['vid_len'],
                                           img_dict['vid_len'])
        except (ValueError, FileNotFoundError):
            return self._rand_another(idx)
        data_dict['data_idx'] = idx
        return data_dict


def build_phyre_dataset(params, val_only=False):
    """Build PHYRE video dataset."""
    args = dict(
        data_root=params.data_root,
        split='val',
        phyre_transform=BaseTransforms(params.resolution),
        seq_size=params.n_sample_frames,
        frame_offset=params.frame_offset,
        fps=params.fps,
        protocal=params.phyre_protocal,
        fold=params.phyre_fold,
        vid_len=params.video_len,
        ratio=params.data_ratio,
        pos_ratio=params.pos_ratio,
        reverse_color=params.reverse_color,
    )
    val_dataset = PHYREDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = PHYREDataset(**args)
    return train_dataset, val_dataset


def build_phyre_slots_dataset(params, val_only=False):
    """Build PHYRE video dataset with pre-computed slots."""
    args = dict(
        data_root=params.data_root,
        slot_root=params.slots_root.format('val'),
        split='val',
        phyre_transform=BaseTransforms(params.resolution),
        seq_size=params.n_sample_frames,
        frame_offset=params.frame_offset,
        fps=params.fps,
        protocal=params.phyre_protocal,
        fold=params.phyre_fold,
        vid_len=params.video_len,
        ratio=params.data_ratio,
        pos_ratio=params.pos_ratio,
        reverse_color=params.reverse_color,
        load_img=params.loss_dict['use_img_recon_loss'],
    )
    val_dataset = PHYRESlotsDataset(**args)
    val_dataset.load_img = True  # to eval img_recon loss
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['slot_root'] = params.slots_root.format('train')
    train_dataset = PHYRESlotsDataset(**args)
    return train_dataset, val_dataset


def build_phyre_rollout_slots_dataset(params, val_only=False):
    """Build PHYRE video dataset with slots rollout by SlotFormer."""
    args = dict(
        data_root=params.data_root,
        slot_root=params.slot_root.format('val'),
        split='val',
        phyre_transform=BaseTransforms(params.resolution),
        seq_size=params.n_sample_frames,
        frame_offset=params.frame_offset,
        fps=params.fps,
        protocal=params.phyre_protocal,
        fold=params.phyre_fold,
        vid_len=params.video_len,
        ratio=params.data_ratio,
        pos_ratio=params.pos_ratio,
        reverse_color=params.reverse_color,
        load_img=False,
    )
    val_dataset = PHYRESlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['slot_root'] = params.slot_root.format('train')
    train_dataset = PHYRESlotsDataset(**args)
    return train_dataset, val_dataset
