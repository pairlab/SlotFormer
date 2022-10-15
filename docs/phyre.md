# PHYRE

We experiment on planning task in this dataset.

Note that, PHYRE follows a 10-fold evaluation protocol.
Below we provide instructions on `fold0`, which can be easily extended to other folds by modifying the `phyre_fold` value in the config file.

## Pre-train SAVi on PHYRE videos

Run the following command to train SAVi on PHYRE videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/savi_phyre_params-fold0.py --fp16 --ddp --cudnn
```

We provide pre-trained SAVi weight as `pretrained/savi_phyre_params-fold0/model_30.pth`.

Then, we'll need to extract slots and save them.
Please use [extract_phyre_slots.py](../slotformer/base_slots/extract_phyre_slots.py) and run:

```
python slotformer/base_slots/extract_phyre_slots.py --params slotformer/base_slots/configs/savi_phyre_params-fold0.py --weight $WEIGHT --save_path $SAVE_PATH (e.g. './data/PHYRE') --vid_len 11 --split -1
```

This will extract slots from PHYRE videos, and save them as `.npy` files under `$SAVE_PATH/slots/savi_phyre_params-fold0/$SETTING`.

-   `--vid_len 11` means we will extract slots up to 11 timesteps, this is because later when training SlotFormer, we will only rollout till timestep 11
-   `--split -1` means we don't parallel slot extraction.
    In fact, since PHYRE dataset is large, extracting slots using single process will take very long time.
    Therefore, you can manually parallelize it by specifying `--total_split` (e.g. 10) and `--split` (0, 1, ..., 9).
    We also provide a script to parallelize it automatically if you use Slurm.
    Take a look at [parallel_phyre.sh](../slotformer/base_slots/parallel_phyre.sh)

## Planning

For the action planning task, we follow `RPIN` to do model-based planning by treating SlotFormer as a world model.
Specifically, we follow the below steps:

-   Train SlotFormer on slots
-   Unroll slots
-   Train a task success classifier (cls) on unrolled slots
-   Plan actions on the test set using trained SlotFormer, and rank them using trained cls

### Train SlotFormer on PHYRE slots

Train a SlotFormer model on extracted slots by running:

```
python scripts/train.py --task video_prediction --params slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py --fp16 --cudnn
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/slotformer_phyre_params-fold0/model_50.pth`.

### Unroll SlotFormer for planning task

To unroll videos, please use [rollout_phyre_slots.py](../slotformer/video_prediction/rollout_phyre_slots.py) and run:

```
python slotformer/video_prediction/rollout_phyre_slots.py --params slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py --weight $WEIGHT --save_path $SAVE_PATH (e.g. './data/PHYRE') --vid_len 11 --split -1
```

This will unroll slots from PHYRE videos, and save them as `.npy` files under `$SAVE_PATH/slots/slotformer_phyre_params-fold0/$SETTING`.

Again, you can parallelize this process by setting different `--split`, or use the provided [parallel_phyre.sh](../slotformer/base_slots/parallel_phyre.sh) script.

### Train task success classifier

TODO:

### Evaluate planning results

TODO:
