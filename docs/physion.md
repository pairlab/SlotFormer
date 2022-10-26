# Physion

We experiment on VQA task in this dataset.

## Pre-train STEVE on Physion videos

STEVE training involves 2 steps: first train a dVAE to discretize images into patch tokens, and then train STEVE to reconstruct these tokens.

### Train dVAE

Run the following command to train dVAE on Physion videos.

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/dvae_physion_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained dVAE weight** as `pretrained/dvae_physion_params/model_20.pth`.

Then, you can choose to extract tokens and save them to disk, reducing the training time of STEVE.
To do so, please use [tokenize_images.py](../slotformer/base_slots/tokenize_images.py) and run:

```
python slotformer/base_slots/tokenize_images.py \
    --params slotformer/base_slots/configs/dvae_physion_params.py \
    --weight $WEIGHT
```

This will save the tokens as `.npy` files under `$DATA_ROOT/Physion/xxxNpys-dvae_physion_params/`.
We cannot provide the tokens because they are too large.

### Train STEVE

Run the following command to train STEVE on Physion video tokens.
You only need to launch 1 run because STEVE training is quite stable (but it takes lots of GPU memory).

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/steve_physion_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained STEVE weight** as `pretrained/steve_physion_params/model_10.pth`.

Then, we'll need to extract slots and save them.
Please use [extract_slots.py](../slotformer/base_slots/extract_slots.py) and run:

```
python slotformer/base_slots/extract_slots.py \
    --params slotformer/base_slots/configs/steve_physion_params.py \
    --weight $WEIGHT \
    --subset $SUBSET --save_path $SAVE_PATH (e.g. './data/Physion/$SUBSET_slots.pkl')
```

Since there are 3 subsets in Physion dataset, namely, `training`, `readout`, `test`, you'll need to extract slots from all of them (16G, 7.8G, 1.2G).

## VQA

For the VQA task, we follow the official benchmark protocol as:

-   Train a dynamics model (SlotFormer) using `training` subset slots
-   Unroll slots on `readout` and `test` subset
-   Train a linear readout model on the unrolled `readout` subset slots + GT labels
-   Evaluate the linear readout model on the unrolled `test` subset slots + GT labels

### Train SlotFormer on Physion slots

Train a SlotFormer model on extracted slots by running:

```
python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_physion_params.py \
    --fp16 --cudnn
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/slotformer_physion_params/model_25.pth`.

### Unroll SlotFormer for VQA task

To unroll videos, please use [rollout_physion_slots.py](../slotformer/video_prediction/rollout_physion_slots.py) and run:

```
python slotformer/video_prediction/rollout_physion_slots.py \
    --params slotformer/video_prediction/configs/slotformer_physion_params.py \
    --weight $WEIGHT \
    --subset $SUBSET --save_path $SAVE_PATH (e.g. './data/Physion/rollout_$SUBSET_slots.pkl')
```

This will unroll slots for Physion videos, and save them into a `.pkl` file.
Please unroll for both `readout` and `test` subset.

### Train linear readout model

Train a linear readout model on rollout slots in the `readout` subset by running:

```
python scripts/train.py --task physion_vqa \
    --params slotformer/physion_vqa/configs/readout_physion_params.py \
    --fp16 --cudnn
```

This will train a readout model that takes in slots extracted from a video, and predict whether two object-of-interests contact during the video.

### Evaluate VQA results

Finally, we can evaluate the trained readout model on rollout slots in the `test` subset, which is the number we report in the paper.
To do this, please use [test_physion_vqa.py](../slotformer/physion_vqa/test_physion_vqa.py) and run:

```
python slotformer/physion_vqa/test_physion_vqa.py \
    --params slotformer/physion_vqa/configs/readout_physion_params.py \
    --weight $WEIGHT
```

You can specify a single weight file to test, or a directory.
If the later is provided, we will test all the weights under that directory, and report the best accuracy of all the models tested.
You can also use the `--threshs ...` flag to specify different thresholds for binarizing the logits to 0/1 predictions.
Again, if multiple thresholds are provided, we will test all of them and report the best one.

**Note**: in our experiments, we noticed that the readout accuracy is not very stable.
So we usually train over three random seeds (using `dup_run_sbatch.sh`), and report the best performance among them.
