# PHYRE

We experiment on planning task in this dataset.

## Pre-train SAVi on PHYRE videos

Run the following command to train SAVi on PHYRE videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/savi_phyre_params-fold0.py --fp16 --ddp --cudnn
```

Note that, PHYRE follows a 10-fold evaluation protocol.
So you'll need to **train 10 SAVi**, by modifying the `phyre_fold` value in the config file.
We provide pre-trained SAVi weight on `fold0` as `pretrained/savi_phyre_params-fold0/model_30.pth`.

Then, we'll need to extract slots and save them.
Please go to the folder of [extract_phyre_slots.py](../slotformer/base_slots/extract_phyre_slots.py) and run:

```
python extract_phyre_slots.py --params configs/savi_phyre_params-fold0.py --weight $WEIGHT --vid_len 11 --split -1
```

This will extract slots from PHYRE videos, and save them as `.npy` files under `$DATA_ROOT/PHYRE/slots/savi_phyre_params-fold0/$SETTING`.
We cannot provide the slots because they are too large.

-   `--vid_len 11` means we will extract slots up to 11 timesteps, this is because later when training SlotFormer, we will only rollout till timestep 11
-   `--split -1` means we don't parallel slot extraction.
    In fact, since PHYRE dataset is large, extracting slots using single process will take very long time.
    Therefore, you can manually parallelize it by specifying `--total_split` (e.g. 10) and `--split` (0, 1, ..., 9).
    We also provide a script to parallelize it automatically if you use Slurm.
    Take a look at [parallel_phyre.sh](../slotformer/base_slots/parallel_phyre.sh)

## Planning

### Train SlotFormer on Physion slots

TODO:

### Unroll SlotFormer for planning task

TODO:

### Train task success classifier

TODO:

### Evaluate planning results

TODO:
