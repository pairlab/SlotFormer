# Dataset Preparation

All datasets should be downloaded or soft-linked to `./data/`.
Or you can modify the `data_root` value in the config files.

## OBJ3D

This dataset is adopted from [G-SWM](https://github.com/zhixuan-lin/G-SWM#datasets).
You can download it manually from the [Google drive](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view), or use the script provided in that repo.

Since the videos in OBJ3D are already extracted to frames, we don't need to further process them.

## CLEVRER

Please download CLEVRER from the official [website](http://clevrer.csail.mit.edu/).

-   We will need `Training Videos, Annotations, Validation Videos, Annotations` for all tasks
-   If you want to experiment on the video prediction task, please download `Object Masks and Attributes` as we will evaluate the quality of the predicted object masks and bboxes
-   If you want to experiment on the VQA task, please download `Training/Validation Questions and Answers, Testing Videos, Questions`, as we will train a VQA module on top of slots, and submit the test set results to the evaluation server

To accelerate training, we can extract videos to frames in advance.
Please run `python scripts/data_preproc/clevrer_video2frames.py`.
You can modify a few parameters in that file.

## Physion

Please download Physion from their github [repo](https://github.com/cogtoolslab/physics-benchmarking-neurips2021#downloading-the-physion-dataset).
Specifically, we only need 2 files containing videos and label files.
The HDF5 files containing additional vision data like depth map, segmentation masks are not needed.

-   Download `PhysionTest-Core` (the 270 MB one) with the [link](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip), and unzip it to a folder named `PhysionTestMP4s`
-   Download `PhysionTrain-Dynamics` (the 770 MB one) with the [link](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/PhysionTrainMP4s.tar.gz), and unzip it to a folder named `PhysionTrainMP4s`
-   Download the labels for the readout subset [here](https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/data/readout_labels.csv), and put it under `PhysionTrainMP4s`

Again we want to extract frames from videos.
We extract all the videos under `PhysionTrainMP4s/` and `PhysionTestMP4s/*/mp4s-redyellow/`.
Please run the provided script `python scripts/data_preproc/physion_video2frames.py`.
You can modify a few parameters in that file as well.

## PHYRE

PHYRE data are simulated on-the-fly, just create an empty directory at the beginning, and make sure to install the `phyre` package via `pip`.

**The `data` directory should look like this:**

```
data/
├── OBJ3D/
│   ├── test/
│   ├── train/
│   └── val/
├── CLEVRER
│   ├── annotations/
│   ├── derender_proposals/  # object masks, used in video prediction
│   ├── questions/  # question and answers, used in VQA
│   ├── videos/
│   │   ├── test/  # test videos are only used in VQA
│   │   ├── train/
│   └   └── val/
├── Physion
│   ├── PhysionTestMP4s/
│   │   ├── Collide/  # 8 scenarios
│   │   ├── Contain/
•   •   •
•   •   •
│   │   ├── Support/
│   │   └── labels.csv  # test subset labels
│   ├── PhysionTrainMP4s/
│   │   ├── Collide_readout_MP4s/  # 8 scenarios x 2 subsets (training, readout)
│   │   ├── Collide_training_MP4s/
•   •   •
•   •   •
│   │   ├── Support_readout_MP4s/
│   │   ├── Support_training_MP4s/
│   └   └── readout_labels.csv  # readout subset labels
├── PHYRE/  # an empty directory
```
