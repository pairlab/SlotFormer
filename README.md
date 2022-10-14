# SlotFormer

This is the official PyTorch implementation for paper: [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861).
The code contains:

-   Training base object-centric slot models
-   Video prediction task on OBJ3D and CLEVRER
-   VQA task on CLEVRER
-   VQA task on Physion
-   Planning task on PHYRE

## Update

-   2022.11: Initial code release!

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

We use [wandb](https://wandb.ai/) for logging, please run `wandb login` to log in.

## Experiments

**This codebase is tailored to [Slurm](https://slurm.schedmd.com/documentation.html) GPU clusters with preemption mechanism.**
For the configs, we mainly use RTX6000 with 24GB memory (though many experiments don't require so much memory).
Please modify the code accordingly if you are using other hardware settings:

-   Please go through `scripts/train.py` and change the fields marked by `TODO:`
-   Please read the config file for the model you want to train.
    We use DDP with multiple GPUs to accelerate training.
    You can use less GPUs to achieve a better memory-speed trade-off

### Dataset Preparation

Please refer to [data.md](docs/data.md) for steps to download and pre-process each dataset.

### Pre-trained Models & Intermediate Data

We provide some pre-trained weights and generated data (e.g. slots) used in our experiments.
However, some data require you to re-generate using our scripts, because they are too large to upload.

Please download the pre-trained weights from [Google drive]() and unzip them to `./pretrained/`.

Please download [OBJ3D slots]() and [CLEVRER slots](), and put them under `./data/OBJ3D/` and `./data/CLEVRER/`, respectively.

### Reproduce Results

Please see [benchmark.md](docs/benchmark.md) for detailed instructions on how to reproduce our results in the paper.

## Citation

Please cite our paper if you find it useful in your research:

```
@article{wu2022slotformer,
  title={SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models},
  author={Wu, Ziyi and Dvornik, Nikita and Greff, Klaus and Kipf, Thomas and Garg, Animesh},
  journal={arXiv preprint arXiv:2210.05861},
  year={2022}
}
```

## Acknowledgement

We thank the authors of [RPIN](https://github.com/HaozhiQi/RPIN) and [Aloe](https://github.com/deepmind/deepmind-research/tree/master/object_attention_for_reasoning) for opening source their wonderful works.

## License

SlotFormer is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Ziyi Wu dazitu616@gmail.com
