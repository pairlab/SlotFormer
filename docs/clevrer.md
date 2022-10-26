# CLEVRER

We experiment on video prediction and VQA task in this dataset.

## Pre-train SAVi on CLEVRER videos

Run the following command to train SAVi on CLEVRER videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained SAVi weight** as `pretrained/stosavi_clevrer_params/model_12.pth`.

Then, we'll need to extract slots and save them.
Please use [extract_slots.py](../slotformer/base_slots/extract_slots.py) and run:

```
python slotformer/base_slots/extract_slots.py \
    --params slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/CLEVRER/slots.pkl')
```

This will extract slots from CLEVRER videos, and save them into a `.pkl` file (~13G).

Alternatively, we also provide pre-computed slots as described in [benchmark.md](./benchmark.md).

## Video prediction

For the video prediction task, we train SlotFormer over slots, and then evaluate the generated frames' visual quality, and object trajectories (mask/bbox).

### Train SlotFormer on CLEVRER slots

Train a SlotFormer model on extracted slots by running:

```
python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/slotformer_clevrer_params/model_80.pth`.

### Evaluate video prediction results

To evaluate the video prediction task, please use [test_vp.py](../slotformer/video_prediction/test_vp.py) and run:

```
python slotformer/video_prediction/test_vp.py \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --weight $WEIGHT
```

This will compute and print all the metrics.
Besides, it will also save 10 videos for visualization under `vis/clevrer/$PARAMS/`.
If you only want to do visualizations (i.e. not testing the metrics), simply use the `--save_num` args and set it to a positive value.

## VQA

For the VQA task, we leverage the SlotFormer model trained above.
We explicitly unroll videos to future frames, and provide them as inputs to train the downstream VQA task model (`Aloe`).

### Unroll SlotFormer for VQA task

To unroll videos, please use [rollout_clevrer_slots.py](../slotformer/video_prediction/rollout_clevrer_slots.py) and run:

```
python slotformer/video_prediction/rollout_clevrer_slots.py \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/CLEVRER/rollout_slots.pkl')
```

This will unroll slots for CLEVRER videos, and save them into a `.pkl` file (~16G).

Alternatively, we provide rollout slots as described in [benchmark.md](./benchmark.md).

### Train Aloe VQA model

To train an Aloe model only on the observed slots (this is the baseline in our paper), run:

```
python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params.py \
    --fp16 --ddp --cudnn
```

To train an Aloe model on both observed and explicitly unrolled slots, run:

```
python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --fp16 --ddp --cudnn
```

All the settings, except the slots are the same in these two experiments.
Alternatively, we also provide **pre-trained Aloe weight** as `pretrained/aloe_clevrer_params-rollout/model_400.pth`.

### Evaluate VQA results

Finally, to evaluate the VQA model on the test set, please use [test_clevrer_vqa.py](../slotformer/clevrer_vqa/test_clevrer_vqa.py) and run:

```
python slotformer/clevrer_vqa/test_clevrer_vqa.py \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --weight $WEIGHT
```

This will save the results as `CLEVRER.json` under the same directory as the weight (we attach our result file as `pretrained/aloe_clevrer_params-rollout/CLEVRER.json`).
You can submit it to the [evaluation server](https://eval.ai/web/challenges/challenge-page/667/overview) of CLEVRER as see the results.
