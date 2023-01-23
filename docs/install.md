# Install

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup:

```
conda create -n slotformer python=3.8.8
conda activate slotformer
```

Then install PyTorch which is compatible with your cuda setting.
In our experiments, we use PyTorch 1.10.1 and CUDA 11.3:

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
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
pip install pycocotools scikit-image lpips
pip install einops==0.3.2  # tested on 0.3.2, other versions might also work
pip install phyre==0.2.2  # please use the v0.2.2, since the task split might slightly differs between versions
```

Finally, clone and install this project by:

```
cd ..  # move out from nerv/
git clone git@github.com:pairlab/SlotFormer.git
cd SlotFormer
pip install -e .
```

We use [wandb](https://wandb.ai/) for logging, please run `wandb login` to log in.

## Possible Issues

-   In case you encounter any environmental issues, you can refer to the conda env file exported from my server [environment.yml](../environment.yml).
    You can install the same environment by `conda env create -f environment.yml`.
