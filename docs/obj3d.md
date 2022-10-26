# OBJ3D

We experiment on video prediction task in this dataset.

## Pre-train SAVi on OBJ3D videos

Run the following command to train SAVi on OBJ3D videos.
Please launch 3 runs and select the best model weight.

```
python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained SAVi weight** as `pretrained/savi_obj3d_params/model_40.pth`.

Then, we'll need to extract slots and save them.
Please use [extract_slots.py](../slotformer/base_slots/extract_slots.py) and run:

```
python slotformer/base_slots/extract_slots.py \
    --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/OBJ3D/slots.pkl')
```

This will extract slots from OBJ3D videos, and save them into a `.pkl` file (~692M).

Alternatively, we also provide pre-computed slots as described in [benchmark.md](./benchmark.md).

## Video prediction

For the video prediction task, we train SlotFormer over slots, and then evaluate the generated frames' visual quality.

### Train SlotFormer on OBJ3D slots

Train a SlotFormer model on extracted slots by running:

```
python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_obj3d_params.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/slotformer_obj3d_params/model_200.pth`.

### Evaluate SlotFormer in video prediction

To evaluate the video prediction task, please use [test_vp.py](../slotformer/video_prediction/test_vp.py) and run:

```
python slotformer/video_prediction/test_vp.py \
    --params slotformer/video_prediction/configs/slotformer_obj3d_params.py \
    --weight $WEIGHT
```

This will compute and print all the metrics.
Besides, it will also save 10 videos for visualization under `vis/obj3d/$PARAMS/`.
If you only want to do visualizations (i.e. not testing the metrics), simply use the `--save_num` args and set it to a positive value.
