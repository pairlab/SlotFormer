# Benchmark

## Overview

The basic experiment pipeline in this project are:

1. Pre-train object-centric slot models (`SAVi` or `STEVE`) on raw videos.
   After training, the models should be able to decompose the scene into meaningful objects, represented by a set of slots
2. Apply pre-trained object-centric models to extract slots from videos, and save them to disk
3. Train `SlotFormer` over the extracted slots to learn the dynamics of videos
4. Evaluate the learned dynamics, which differs between tasks:

-   In video prediction, directly evaluate the predicted frames and object masks/bboxes
-   In VQA and planning tasks, train downstream task models over extracted and rollout slots

### Pre-trained Models & Intermediate Data

We provide some pre-trained weights and generated data (e.g. slots) used in our experiments [here](https://drive.google.com/drive/folders/15y21miKZsAVHOSQEZLbUBWRrsZzcd5QW?usp=sharing).
However, some data require you to re-generate using our scripts, because they are too large to upload.

Please download the pre-trained weights `pretrained.zip` and unzip them to `./pretrained/`.

If you want to test the video prediction results on OBJ3D and CLEVRER, please download OBJ3D slots `slots/obj3d_slots.pkl`, and CLEVRER slots `slots/clevrer_slots.pkl`, and put them under `./data/OBJ3D/` and `./data/CLEVRER/`, respectively.

If you want to test the VQA results on CLEVRER, please download the CLEVRER slots rollouted by SlotFormer `slots/rollout_clevrer_slots.pkl`, and put it under `./data/CLEVRER/`.

## Basic Usage

**We provide a unified script [train.py](../scripts/train.py) to train all models used in this project.**
You should always call it in the root directory of this repo (i.e. calling `python scripts/train.py xxx`).

**All of the model training can be done by specifying the task it belongs to, providing a config file (called `params` here), and adding other args.**
Please check the config file for the GPUs and other resources (e.g. `num_workers` CPUs) before launching a training.

For example, to train a SAVi model on OBJ3D dataset, simply run:

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/savi_obj3d_params.py
```

Other arguments include:

-   `--weight`: load weights
-   `--fp16`: enable half-precision training
-   `--ddp`: use DDP multi-GPU training
-   `--cudnn`: enable cudnn benchmark
-   `--local_rank`: this one is for DDP, don't change it

**When generating intermediate data (e.g. pre-computed video slots), we usually save them under the same directory as the dataset's root.**
**When producing final results (e.g. video visualizations, VQA result files to submit), we usually save them under the same directory as the model weight used to generate them.**
See the docs of each dataset below for more details.

### Scripts

We provide helper scripts if you're running experiments on a Slurm GPU cluster.

You can use [sbatch_run.sh](../scripts/sbatch_run.sh) to automatically generate a sbatch file and submit the job to slurm.
Simply running:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS \
    ./scripts/sbatch_run.sh $PARTITION $JOB_NAME \
    scripts/train.py ddp (if using 1 GPU then `none`) --py_args...
```

For example, to train a SAVi model on OBJ3D dataset, we can set `--py_args...` as (see the config file for the number of GPU/CPU to use)

```
--task base_slots --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --fp16 --ddp --cudnn
```

Then this will be equivalent to running the following command in CLI:

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --fp16 --ddp --cudnn
```

We also provide a script to **submit multiple runs of the same experiment** to slurm.
This is important because from our experiments, SAVi training is unstable.
If I launch 3 runs with different random seeds, some might succeed in scene decomposition while others could fail.
Therefore, **when training SAVi models, we always run 3 different seeds** and select the weight with the best validation loss and scene decomposition visual quality.
On the other hand, STEVE training is usually stable, so we only train once.

**We note that this is not a limit of our work**.
In this paper, we propose SlotFormer as a dynamics model building upon any base slot models.
So improving the stability of SAVi is beyond the scope of this work.

To use the duplicate-run script [dup_run_sbatch.sh](../scripts/dup_run_sbatch.sh), simply do:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS REPEAT=$NUM_REPEAT \
    ./scripts/dup_run_sbatch.sh $PARTITION $JOB_NAME \
    scripts/train.py ddp $PARAMS --py_args...
```

The other parts are really the same as `sbatch_run.sh`.
The only difference is that we need to input the config file `$PARAMS` separately, so that the script will make several copies to it, and submit different jobs.

Again if we want to train a SAVi model on OBJ3D dataset, with `4` GPUs and `4x8=32` CPUs, duplicating `3` times, on `rtx6000` partition, and in the name of `savi_obj3d_params`, simply run:

```
GPUS=4 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 \
    ./scripts/dup_run_sbatch.sh rtx6000 savi_obj3d_params \
    scripts/train.py ddp slotformer/base_slots/configs/savi_obj3d_params.py \
    --task base_slots --fp16 --ddp --cudnn
```

### Intermediate Data

Our training often involves multiple steps, e.g. first pre-train SAVi, and then train SlotFormer on slots extracted by it.
To save time and computation, we will store these intermediate data to hard disk, whose size ranges from a few hundred MB to 20G.
We provide scripts to do this which will be described in detail later for each task.
For ease of file management, every time when an intermediate data is generated by a trained model (assume its weight is `$DIR/$WEIGHT.pth`), we will soft-link the data to `$DIR/`.
You can modify the code if you want a different behavior.

## OBJ3D

For the video prediction task on OBJ3D dataset, see [obj3d.md](./obj3d.md).

## CLEVRER

For the video prediction and VQA task on CLEVRER dataset, see [clevrer.md](./clevrer.md).

## Physion

For the VQA task on Physion dataset, see [physion.md](./physion.md).

## PHYRE

For the action planning task on PHYRE dataset, see [phyre.md](./phyre.md).
