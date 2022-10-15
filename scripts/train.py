"""A unified training script for all models used in the SlotFormer project."""

import os
import sys
import pwd
import importlib
import argparse
import wandb

import torch

from nerv.utils import mkdir_or_exist
from nerv.training import BaseDataModule


def main(params):
    # build datamodule
    datasets = build_dataset(params)
    train_set, val_set = datasets[0], datasets[1]
    collate_fn = datasets[2] if len(datasets) == 3 else None
    datamodule = BaseDataModule(
        params,
        train_set=train_set,
        val_set=val_set,
        use_ddp=params.ddp,
        collate_fn=collate_fn,
    )

    # build model
    model = build_model(params)

    # create checkpoint dir
    exp_name = os.path.basename(args.params)
    ckp_path = os.path.join('./checkpoint/', exp_name, 'models')
    if args.local_rank == 0:
        mkdir_or_exist(os.path.dirname(ckp_path))

        # on clusters, quota under user dir is usually limited
        # soft link to save the weights in temp space for checkpointing
        # e.g. on our cluster, the temp dir is /checkpoint/$USR/$SLURM_JOB_ID/
        # TODO: modify this if you are not running on clusters
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
        if SLURM_JOB_ID and not os.path.exists(ckp_path):
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                pwd.getpwuid(os.getuid())[0], SLURM_JOB_ID, ckp_path))

        # it's not good to hard-code the wandb id
        # but on preemption clusters, we want the job to resume the same wandb
        # process after resuming training (i.e. drawing the same graph)
        # so we have to keep the same wandb id
        # TODO: modify this if you are not running on preemption clusters
        preemption = True
        if SLURM_JOB_ID and preemption:
            logger_id = logger_name = f'{exp_name}-{SLURM_JOB_ID}'
        else:
            logger_name = exp_name
            logger_id = None
        wandb.init(
            project=params.project,
            name=logger_name,
            id=logger_id,
            dir=ckp_path,
        )

    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
    )

    method.fit(
        resume_from=args.weight, san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SlotFormer training script')
    parser.add_argument('--task', type=str, default='base_slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true', help='half-precision')
    parser.add_argument('--ddp', action='store_true', help='DDP training')
    parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # import `build_dataset/model/method` function according to `args.task`
    print(f'INFO: training model in {args.task} task!')
    task = importlib.import_module(f'slotformer.{args.task}')
    build_dataset = task.build_dataset
    build_model = task.build_model
    build_method = task.build_method

    # load the params
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    params.ddp = args.ddp

    if args.fp16:
        print('INFO: using FP16 training!')
    if args.ddp:
        print('INFO: using DDP training!')
    if args.cudnn:
        torch.backends.cudnn.benchmark = True
        print('INFO: using cudnn benchmark!')

    main(params)
