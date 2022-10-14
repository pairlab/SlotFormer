# CLEVRER

We experiment on video prediction and VQA task in this dataset.

## Pre-train SAVi on CLEVRER videos

Run the following command to train SAVi on CLEVRER videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/stosavi_clevrer_params.py --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained SAVi weight** as `pretrained/stosavi_clevrer_params/model_12.pth`.

Then, we'll need to extract slots and save them.
Please go to the folder of [extract_slots.py](../slotformer/base_slots/extract_slots.py) and run:

```
python extract_slots.py --params configs/stosavi_clevrer_params.py --weight $WEIGHT --save_path $SAVE_PATH (e.g. '$DATA_ROOT/CLEVRER/slots.pkl')
```

This will extract slots from CLEVRER videos, and save them into a `.pkl` file (~13G).

Alternatively, we also provide pre-computed slots which can be downloaded (see [README.md](../README.md)).

## Video prediction

### Train SlotFormer on CLEVRER slots

TODO:

### Evaluate video prediction results

TODO:

## VQA

### Unroll SlotFormer for VQA task

TODO:

### Train Aloe VQA model

TODO:

### Evaluate VQA results

TODO:
