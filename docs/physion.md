# Physion

We experiment on VQA task in this dataset.

## Pre-train STEVE on Physion videos

STEVE training involves 2 steps: first train a dVAE to discretize images into patch tokens, and then train STEVE to reconstruct these tokens.

### Train dVAE

Run the following command to train dVAE on Physion videos.

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/dvae_physion_params.py --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained dVAE weight** as `pretrained/dvae_physion_params/model_20.pth`.

Then, you can choose to extract tokens and save them to disk, reducing the training time of STEVE.
To do so, please go to the folder of [tokenize_images.py](../slotformer/base_slots/tokenize_images.py) and run:

```
python tokenize_images.py --params configs/dvae_physion_params.py --weight $WEIGHT
```

This will save the tokens as `.npy` files under `$DATA_ROOT/Physion/xxxNpys-dvae_physion_params/`.
We cannot provide the tokens because they are too large.

### Train STEVE

Run the following command to train STEVE on Physion video tokens.
You only need to launch 1 run because STEVE training is quite stable (but it takes lots of GPU memory).

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/steve_physion_params.py --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained STEVE weight** as `pretrained/steve_physion_params/model_10.pth`.

Then, we'll need to extract slots and save them.
Please go to the folder of [extract_slots.py](../slotformer/base_slots/extract_slots.py) and run:

```
python extract_slots.py --params configs/steve_physion_params.py --weight $WEIGHT --subset $SUBSET --save_path $SAVE_PATH (e.g. '$DATA_ROOT/Physion/$SUBSET_slots.pkl')
```

Since there are 3 subsets in Physion dataset, namely, `training`, `readout`, `test`, you'll need to extract slots from all of them (16G, 7.8G, 1.2G).

## VQA

### Train SlotFormer on Physion slots

TODO:

### Unroll SlotFormer for VQA task

TODO:

### Train linear readout model

TODO:

### Evaluate VQA results

TODO:
