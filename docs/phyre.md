# PHYRE

We experiment on planning task in this dataset.

Note that, PHYRE follows a 10-fold evaluation protocol.
Below we provide instructions on `fold0`, which can be easily extended to other folds by modifying the `phyre_fold` value in the config file.

## Pre-train SAVi on PHYRE videos

Run the following command to train SAVi on PHYRE videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/savi_phyre_params-fold0.py \
    --fp16 --ddp --cudnn
```

We provide pre-trained SAVi weight as `pretrained/savi_phyre_params-fold0/model_30.pth`.

Then, we'll need to extract slots and save them.
Please use [extract_phyre_slots.py](../slotformer/base_slots/extract_phyre_slots.py) and run:

```
python slotformer/base_slots/extract_phyre_slots.py \
    --params slotformer/base_slots/configs/savi_phyre_params-fold0.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/PHYRE') --vid_len 11 --split -1
```

This will extract slots from PHYRE videos, and save them as `.npy` files under `$SAVE_PATH/slots/savi_phyre_params-fold0/$SETTING`.

-   `--vid_len 11` means we will extract slots up to 11 timesteps, this is because later when training SlotFormer, we will only rollout till timestep 11
-   `--split -1` means we don't parallel slot extraction.
    In fact, since PHYRE dataset is large, extracting slots using single process will take very long time.
    Therefore, you can manually parallelize it by specifying `--total_split` (e.g. 10) and `--split` (0, 1, ..., 9).
    We also provide a script [parallel_phyre.sh](../scripts/parallel_phyre.sh) to parallelize it automatically if you use Slurm.
    An example usage is
    ```
    CPUS_PER_TASK=4 ./scripts/parallel_phyre.sh $PARTITION \
        slotformer/base_slots/extract_phyre_slots.py \
        slotformer/base_slots/configs/savi_phyre_params-fold0.py \
        $WEIGHT \
        5 \
        --save_path $SAVE_PATH --vid_len 11
    ```
    This will automatically run the above python command with `--split` equals to 0, 1, 2, 3, 4, i.e. parallelize the slot extraction by 5 processes.

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
python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py \
    --fp16 --cudnn
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/slotformer_phyre_params-fold0/model_50.pth`.

### Unroll SlotFormer for planning task

To unroll videos, please use [rollout_phyre_slots.py](../slotformer/video_prediction/rollout_phyre_slots.py) and run:

```
python slotformer/video_prediction/rollout_phyre_slots.py \
    --params slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/PHYRE') --vid_len 11 --split -1
```

This will unroll slots from PHYRE videos, and save them as `.npy` files under `$SAVE_PATH/slots/slotformer_phyre_params-fold0/$SETTING`.

Again, you can parallelize this process by setting different `--split`, or use the provided [parallel_phyre.sh](../scripts/parallel_phyre.sh) script.

### Train task success classifier

Train a task success classifier on rollout slots by running:

```
python scripts/train.py --task phyre_planning \
    --params slotformer/phyre_planning/configs/readout_phyre_params-fold0.py \
    --fp16 --cudnn
```

This will train a Transformer-based binary classifier on rollout slots, to predict whether the action will lead to success.

### Evaluate planning results

Finally, we can evaluate our models on the test set, which is the number we report in the paper.
We will need three models in this evaluation:

-   SAVi: extract slots from the initial frame (with different actions applied, i.e. placing the red ball with varying sizes at different locations)
-   SlotFormer: rollout the slots from frame 0 to predict the scene dynamics
-   Task success classifier: to predict whether the action will succeed, used in ranking all the candidate actions

To do this, please use [test_phyre_planning.py](../slotformer/phyre_planning/test_phyre_planning.py) and run:

```
python slotformer/phyre_planning/test_phyre_planning.py \
    --params slotformer/video_prediction/configs/slotformer_phyre_params-fold0.py \
    --weight $SlotFormer_WEIGHT \
    --task_cls_params slotformer/phyre_planning/configs/readout_phyre_params-fold0.py \
    --task_cls_weight $CLS_WEIGHT \
    --savi_params slotformer/base_slots/configs/savi_phyre_params-fold0.py \
    --savi_weight $SAVi_WEIGHT \
    --split -1
```

Again, you can parallelize this process by setting different `--split`, or use the provided [parallel_phyre.sh](../scripts/parallel_phyre.sh) script.

The predicted success rate for all tasks and actions will be saved under `os.path.dirname($CLS_WEIGHT)/test/`.
To merge them for computing the AUCCESS metric, simply run:

```
python slotformer/phyre_planning/test_phyre_planning.py \
    --collect os.path.dirname($CLS_WEIGHT)/test/ --total_split $NUM
```

**Note**:

-   Please select the best performing cls weight by looking at the `wandb` logs.
    Usually you can choose the checkpoint with the highest `val/acc_0.50` metric.
    Similar to Physion, the readout model accuracy is also unstable.
    So we usually train over three random seeds and select the one with the highest `val/acc_0.50` value
-   The number reported in the paper is averaged over all 10 folds.
    So you will need to repeat the above process on each fold
