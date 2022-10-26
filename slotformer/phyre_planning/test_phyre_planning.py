import os
import sys
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import phyre
import torch
from torch.utils.data import Dataset, DataLoader

from models import build_model

from slotformer.base_slots import build_model as build_savi_model
from slotformer.video_prediction import build_model as build_slotformer_model
from slotformer.base_slots.datasets.phyre import observations_to_uint8_rgb, \
    BaseTransforms

INVALID = phyre.SimulationStatus.INVALID_INPUT
SUCCESS = phyre.SimulationStatus.SOLVED
FAILED = phyre.SimulationStatus.NOT_SOLVED


class PHYREDataset(Dataset):
    """Reading the first frame for testing PHYRE planning."""

    def __init__(
        self,
        data_root,
        phyre_transform,
        protocal='within',
        fold=0,
        vid_len=15,
        reverse_color=False,
        start_idx=None,
        end_idx=None,
    ):

        self.data_root = data_root
        self.phyre_transform = phyre_transform

        self.protocal = protocal
        self.fold = fold
        self.vid_len = vid_len
        self.reverse_color = reverse_color

        self.start_idx = start_idx
        self.end_idx = end_idx

        self._filter_actions()

    def _preproc_img(self, img):
        img = observations_to_uint8_rgb(img, reverse=self.reverse_color)
        return np.ascontiguousarray(img)

    def _read_frames(self, idx):
        """Read init frame."""
        task_id, act_id = idx // self.num_acts, idx % self.num_acts
        act_label = self.sim_statuses[task_id][act_id]
        # we don't need to process INVALID actions
        if act_label == INVALID:
            img = torch.zeros((3, *self.phyre_transform.resolution)).float()
        else:
            acts = self.act_lst[act_id]
            sim = self.simulator.simulate_action(
                int(task_id),
                acts,
                stride=60,
                need_images=True,
                need_featurized_objects=False,
            )
            assert sim.status == self.sim_statuses[task_id][act_id]
            img = sim.images[0]
            img = self.phyre_transform(self._preproc_img(img)).float()
        img = img.unsqueeze(0)  # expand time dimension
        data_dict = {
            'img': img,
            'task_id': task_id,
            'act_id': act_id,
            'act_label': act_label,
        }
        return data_dict

    def __getitem__(self, idx):
        idx = idx if self.start_idx is None else idx + self.start_idx
        return self._read_frames(idx)

    def __len__(self):
        if self.start_idx is not None:
            return self.end_idx - self.start_idx
        return self.num_tasks * self.num_acts

    def _filter_actions(self):
        """Filter actions that generate videos longer than a length."""
        protocal = self.protocal
        fold = self.fold

        print(f'testing using protocal {protocal} and fold {fold}')

        # setup the PHYRE evaluation split
        eval_setup = f'ball_{protocal}_template'
        _, _, tasks = phyre.get_fold(eval_setup, fold)  # PHYRE setup
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        # filter tasks
        candidate_lst = [f'{i:05d}' for i in range(0, 25)]  # filter tasks
        tasks = [task for task in tasks if task.split(':')[0] in candidate_lst]
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.simulator = phyre.initialize_simulator(tasks, action_tier)

        # the action candidates are provided by the author of PHYRE benchmark
        self.num_acts = 10000
        cache = phyre.get_default_100k_cache('ball')
        self.act_lst = cache.action_array[:self.num_acts]
        training_data = cache.get_sample(tasks, None)
        # (num_tasks x num_actions), whether success when taking that action
        self.sim_statuses = np.array(training_data['simulation_statuses'])
        assert len(self.sim_statuses) == self.num_tasks


@torch.no_grad()
def test(model, savi, dataset):
    """Returns slots extracted from each video of the dataset."""
    torch.cuda.empty_cache()
    total_num = len(dataset)
    if args.split != -1:
        start_idx = total_num // args.total_split * args.split
        end_idx = total_num // args.total_split * (args.split + 1) if \
            args.split < (args.total_split - 1) else total_num
        dataset.start_idx = start_idx
        dataset.end_idx = end_idx
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.cpus,
        pin_memory=True,
        drop_last=False,
    )

    # predicted success confidence, gt success status
    # for each task, we will predict the success probs for all 10k actions
    # then we rank the actions by the predicted probs, and compute AUCCESS
    all_pred_conf = np.ones((dataset.num_tasks, dataset.num_acts)) * -100.
    all_gt_status = np.ones((dataset.num_tasks, dataset.num_acts)) * -100.
    for batch_data in tqdm(dataloader):
        # filter our INVALID actions
        act_label = batch_data['act_label']

        # in case all are INVALID
        if (act_label == INVALID).all().item():
            task_id = batch_data['task_id'].cpu().numpy()
            act_id = batch_data['act_id'].cpu().numpy()
            all_pred_conf[task_id, act_id] = np.ones((act_label.shape[0])) * -1
            all_gt_status[task_id, act_id] = act_label.cpu().numpy()
            torch.cuda.empty_cache()
            continue

        # use SAVi to extract slots of the first frame
        input_data = {
            'img': batch_data['img'][act_label != INVALID].float().cuda(),
        }
        out_dict = savi(input_data)
        slot0 = out_dict['post_slots']  # [B, 1, N, C]

        # SlotFormer rollout based on slot0
        # then apply the task success classifier
        # pad to desired len
        B, _, N, C = slot0.shape
        slots = torch.zeros((B, dataset.vid_len, N, C)).type_as(slot0)
        slots[:, :1] = slot0
        input_data = {'slots': slots}
        out_dict = model(input_data)
        pred_conf = torch.sigmoid(out_dict['logits']).cpu().numpy()
        # pad INVALID with -1 conf
        act_label = act_label.cpu().numpy()
        pad_pred_conf = np.ones((act_label.shape[0])) * -1.
        pad_pred_conf[act_label != INVALID] = pred_conf
        # insert
        task_id = batch_data['task_id'].cpu().numpy()
        act_id = batch_data['act_id'].cpu().numpy()
        all_pred_conf[task_id, act_id] = pad_pred_conf
        all_gt_status[task_id, act_id] = act_label

        torch.cuda.empty_cache()

    # save results
    # if we're doing multi-split parallel testing, we will save the results to
    # temp files, and merge them using `collect_results()` later
    save_path = os.path.join(os.path.dirname(args.task_cls_weight), 'test')
    os.makedirs(save_path, exist_ok=True)
    np.save(
        os.path.join(save_path, f'pred_conf-{args.split}.npy'), all_pred_conf)
    np.save(
        os.path.join(save_path, f'gt_status-{args.split}.npy'), all_gt_status)


def collect_results():
    """Collect the saved npy files."""
    save_path = args.collect
    conf0 = np.load(os.path.join(save_path, 'pred_conf-0.npy'))
    gt0 = np.load(os.path.join(save_path, 'gt_status-0.npy'))
    for split in range(1, args.total_split):
        conf = np.load(os.path.join(save_path, f'pred_conf-{split}.npy'))
        gt = np.load(os.path.join(save_path, f'gt_status-{split}.npy'))
        conf0[gt != -100] = conf[gt != -100]
        gt0[gt != -100] = gt[gt != -100]
    all_conf, all_gt = conf0, gt0
    assert (all_gt != -100.).all() and (all_conf != -100.).all()
    np.save(os.path.join(save_path, 'all_conf.npy'), all_conf)
    np.save(os.path.join(save_path, 'all_gt.npy'), all_gt)
    # auccess
    num_tasks = all_gt.shape[0]
    auccess = np.zeros((num_tasks, 100))
    for task_id in range(num_tasks):
        conf = all_conf[task_id]
        gt = all_gt[task_id]
        # filter out invalid actions
        conf = conf[gt != INVALID]
        gt = gt[gt != INVALID]
        gt[gt == FAILED] = 0  # from -1 to 0 so that it won't decrease the aucc
        top_acc = np.array(gt)[np.argsort(conf)[::-1]]
        for i in range(100):
            auccess[task_id, i] = int(np.sum(top_acc[:i + 1]) > 0)
    w = np.array([np.log(k + 1) - np.log(k) for k in range(1, 101)])
    s = auccess.sum(0) / auccess.shape[0]
    print('Success rate in the first 100 attempts:\n', s)
    print(f'AUCCESS = {np.sum(w * s) / np.sum(w) * 100.:.2f}')


def main():
    assert torch.cuda.device_count() == 1, 'only support single GPU'
    model = build_slotformer_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    # task success classifier
    task_cls = build_model(task_cls_params)
    task_cls.load_state_dict(
        torch.load(args.task_cls_weight, map_location='cpu')['state_dict'])
    model.success_cls = task_cls
    model.use_cls_loss = True
    model = model.eval().cuda()

    savi = build_savi_model(savi_params)
    savi.load_state_dict(
        torch.load(args.savi_weight, map_location='cpu')['state_dict'])
    savi.testing = True  # only extract slots
    savi = savi.eval().cuda()

    test_set = PHYREDataset(
        data_root=params.data_root,
        phyre_transform=BaseTransforms(params.resolution),
        protocal=params.phyre_protocal,
        fold=params.phyre_fold,
        vid_len=params.n_sample_frames,
        reverse_color=params.reverse_color,
    )
    test(model, savi, test_set)


def process_params(params):
    """Import the params class."""
    if params.endswith('.py'):
        params = params[:-3]
    sys.path.append(os.path.dirname(params))
    params = importlib.import_module(os.path.basename(params))
    params = params.SlotFormerParams()
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PHYRE planning')
    parser.add_argument('--params', type=str, default='')
    parser.add_argument(
        '--weight', type=str, default='', help='pretrained SlotFormer weight')
    parser.add_argument('--task_cls_params', type=str, default='')
    parser.add_argument(
        '--task_cls_weight',
        type=str,
        default='',
        help='pretrained task success classifier weight',
    )
    parser.add_argument('--savi_params', type=str, default='')
    parser.add_argument(
        '--savi_weight', type=str, default='', help='pretrained SAVi weight')
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--total_split', type=int, default=10)
    parser.add_argument(
        '--collect', type=str, default='', help='path to npy files')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--cpus', type=int, default=8)
    args = parser.parse_args()

    # collect and evaluate saved results
    if args.collect:
        collect_results()
        exit(-1)

    params = process_params(args.params)
    params.loss_dict['use_img_recon_loss'] = False
    task_cls_params = process_params(args.task_cls_params)
    savi_params = process_params(args.savi_params)

    # adjust rollout len according to task_cls
    vid_len = max(task_cls_params.readout_dict['sel_slots']) + 1
    params.video_len = vid_len * params.fps
    params.n_sample_frames = params.video_len
    params.loss_dict['rollout_len'] = params.video_len - 1

    torch.backends.cudnn.benchmark = True
    main()
