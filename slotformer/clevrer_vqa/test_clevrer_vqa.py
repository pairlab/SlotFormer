"""Testing script for the CLEVRER VQA task."""

import os
import sys
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from nerv.utils import dump_obj

from models import build_model
from datasets import build_dataset


def bool2str(v):
    assert v in [True, False]
    if v:
        return 'correct'
    else:
        return 'wrong'


def int2str(v):
    return str(label2answer[v])


@torch.no_grad()
def test(model, test_loader):
    results = [{
        'scene_index': i + 15000,
        'questions': []
    } for i in range(5000)]
    for data_dict in tqdm(test_loader):
        scene_index = data_dict['scene_index'].numpy().astype(np.int32)  # [B]
        question_id = data_dict['question_id'].numpy().astype(np.int32)  # [B]
        mc_choice_id = data_dict['mc_choice_id'].numpy().astype(np.int32)
        mc_flag = data_dict['mc_flag'].numpy().astype(np.int32)  # both [B2 n]
        out_dict = model({k: v.to(model.device) for k, v in data_dict.items()})
        # [B1, num_cls], [B2 num_choices]
        cls_answer_logits = out_dict['cls_answer_logits']
        if cls_answer_logits is not None:
            cls_answer = cls_answer_logits.argmax(-1).cpu().numpy()  # [B1]
        mc_answer_logits = out_dict['mc_answer_logits']
        if mc_answer_logits is not None:
            mc_answer = (mc_answer_logits > 0.).cpu().numpy()  # [B2 n]
        num_cls = cls_answer_logits.shape[0] if \
            cls_answer_logits is not None else 0
        num_mc = mc_flag.max().item() + 1 if \
            mc_answer_logits is not None else 0

        # based on the fact that cls q are always before mc q in loaded data
        for i in range(num_cls):
            idx = i
            res_idx = scene_index[idx] - 15000
            q_id = question_id[idx]
            ans = cls_answer[idx]
            results[res_idx]['questions'].append({
                'question_id': int(q_id),
                'answer': int2str(int(ans)),
            })

        for i in range(num_mc):
            idx = i + num_cls
            res_idx = scene_index[idx] - 15000
            q_id = question_id[idx]
            ans = mc_answer[mc_flag == i]  # [n]
            choice_id = mc_choice_id[mc_flag == i]
            choice_lst = [{
                'choice_id': int(choice_id[j]),
                'answer': bool2str(ans[j]),
            } for j in range(len(choice_id))]
            q_list = results[res_idx]['questions']
            flag = None
            for j, lst in enumerate(q_list):
                if lst['question_id'] == q_id:
                    flag = j
                    break
            if flag is None:
                q_list.append({
                    'question_id': int(q_id),
                    'choices': choice_lst,
                })
            else:
                q_list[flag]['choices'] += choice_lst
    save_path = os.path.join(os.path.dirname(args.weight), 'CLEVRER.json')
    dump_obj(results, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aloe CLEVRER VQA')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()

    test_set, collate_fn = build_dataset(params, test_set=True)
    label2answer = test_set.label2answer
    test_loader = DataLoader(
        test_set,
        params.val_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(params)
    ckp = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(ckp['state_dict'])
    model = model.eval().cuda()

    test(model, test_loader)
