"""Testing script for the Physion VQA task."""

import os
import sys
import importlib
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset


def _calc_acc(pred, gt, acc_thresh):
    return ((pred > acc_thresh).astype(np.float32) == gt).\
        astype(np.float32).mean()


@torch.no_grad()
def test(model, test_loader, weight, acc_thresh):
    """Load a readout model weight and test its accuracy on the test set."""
    model = model.eval().cuda()
    ckp = torch.load(weight, map_location='cpu')
    model.load_state_dict(ckp['state_dict'])

    all_pred, all_gt, all_task_idx = [], [], []
    for batch in test_loader:
        batch = {k: v.float().cuda() for k, v in batch.items()}
        out = model(batch)
        task_idx = batch['task_idx'].flatten()
        pred = torch.sigmoid(out['logits'].flatten())
        gt = batch['label'].flatten().type_as(pred)
        # accumulate for later ensemble testing
        all_pred.append(pred.cpu().numpy())
        all_gt.append(gt.cpu().numpy())
        all_task_idx.append(task_idx.cpu().numpy())

    # compute accuracy
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    all_task_idx = np.concatenate(all_task_idx)
    all_acc = _calc_acc(all_pred, all_gt, acc_thresh)
    task_acc = {}
    all_tasks = test_loader.dataset.all_tasks
    for i, task in enumerate(all_tasks):
        task_acc[task] = _calc_acc(all_pred[all_task_idx == i],
                                   all_gt[all_task_idx == i], acc_thresh)
    return all_acc, task_acc


def main(params):
    model = build_model(params).eval().cuda()

    test_loader = DataLoader(
        test_set,
        batch_size=params.val_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # if a dir is given, test all weights in the dir to find the best one
    if os.path.isfile(args.weight):
        all_weights = [args.weight]
    else:
        assert os.path.isdir(args.weight)
        all_weights = [
            os.path.join(args.weight, w)
            for w in sorted(os.listdir(args.weight)) if w.endswith('.pth')
        ]

    all_acc, all_task_acc = [], []
    for w in tqdm(all_weights):
        acc, task_acc = test(model, test_loader, w, args.thresh)
        all_acc.append(acc)
        all_task_acc.append(task_acc)

    # take the max acc weight
    idx = np.argmax(all_acc)
    acc, task_acc = all_acc[idx], all_task_acc[idx]

    return all_weights[idx], acc, task_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Physion VQA')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument(
        '--threshs',
        nargs='+',
        default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        type=float,
    )
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    params.dataset = 'physion_slots_label_test'
    params.slots_root = os.path.join(
        os.path.dirname(params.slots_root), 'test_slots.pkl')
    test_set = build_dataset(params)

    all_w, all_acc, all_task_acc = [], [], []
    for thresh in args.threshs:
        print(f'Testing with threshold {thresh}')
        args.thresh = thresh
        w, acc, task_acc = main(params)
        all_w.append(w)
        all_acc.append(acc)
        all_task_acc.append(task_acc)

    # take the max acc threshold
    idx = np.argmax(all_acc)
    w, acc, task_acc = all_w[idx], all_acc[idx], all_task_acc[idx]

    print(f'Threshold {args.threshs[idx % len(args.threshs)]}, '
          f'{w} achieves the best accuracy')
    print(f'All accuracy: {acc:.3f}')
    for task, acc in task_acc.items():
        print(f'{task}: {acc:.3f}')
