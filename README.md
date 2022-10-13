# SlotFormer

This is the official PyTorch implementation for paper: [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861).
The code contains:

-   Training base object-centric slot models
-   Training SlotFormer dynamics models for video prediction task
-   VQA task on CLEVRER
-   VQA task on Physion
-   Planning task on PHYRE

## Update

- 2022.11: Initial code release!

## Prerequisites

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup:

```
conda create -n slotformer python=3.8
conda activate slotformer
```

Then install PyTorch which is compatible with your cuda setting.
In our experiments, we use PyTorch 1.10.1 and CUDA 11.3:

```
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

The codebase heavily relies on [nerv](https://github.com/Wuziyi616/nerv) for project template and Trainer.
You can easily install it by:

```
git clone git@github.com:Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
pip install -e .
```

This will automatically install packages necessary for the project.
Additional packages are listed as follows:

```
# mask evaluation on CLEVRER video prediction task
pip install pycocotools
# read Physion VQA task labels
pip install pandas
# STEVE model used in Physion VQA task
pip install einops==0.3.2  # tested on 0.3.2, other versions might also work
# PHYRE simulator used in PHYRE planning task
pip install phyre==0.2.1  # please use the latest v0.2.1, since the task split slightly differs between versions
```

Finally, install this project by `pip install -e .`

## Experiments

**This codebase is tailored to Slurm GPU clusters with pre-emption mechanism.
For the configs, we mainly use RTX6000 with 24GB memory (though many experiments don't require so much memory).
Please modify the code accordingly if you are using other hardware settings.**


