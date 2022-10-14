import os
from concurrent import futures as futures

from nerv.utils import VideoReader, load_obj

data_root = './data/Physion'
NUM_FRAMES = 200
RESIZE = (128, 128)
NUM_WORKERS = 8

training_train_task2path = load_obj(
    'slotformer/base_slots/datasets/splits/Physion/training_train.json')
training_val_task2path = load_obj(
    'slotformer/base_slots/datasets/splits/Physion/training_val.json')
readout_train_task2path = load_obj(
    'slotformer/base_slots/datasets/splits/Physion/readout_train.json')
readout_val_task2path = load_obj(
    'slotformer/base_slots/datasets/splits/Physion/readout_val.json')
test_task2path = load_obj(
    'slotformer/base_slots/datasets/splits/Physion/test_test.json')
all_tasks = list(training_train_task2path.keys())


def video2frames(video_path):
    video_path = os.path.join(data_root, video_path)
    cap = VideoReader(video_path)
    frame_dir = video_path[:-4]  # '.mp4'
    if os.path.exists(frame_dir):  # already exists
        return
    cap.cvt2frames(frame_dir, target_shape=RESIZE, max_num=NUM_FRAMES)


def process_videos(subset):
    train_task2path, val_task2path = None, None
    if subset == 'test':
        train_task2path = test_task2path
    elif subset == 'training':
        train_task2path = training_train_task2path
        val_task2path = training_val_task2path
    elif subset == 'readout':
        train_task2path = readout_train_task2path
        val_task2path = readout_val_task2path
    else:
        raise NotImplementedError(f'{subset} is not implemented')

    for task in all_tasks:
        print(f'Processing {task} training set...')
        # multi-processing
        with futures.ThreadPoolExecutor(NUM_WORKERS) as executor:
            executor.map(video2frames, train_task2path[task])

        if val_task2path is not None:
            print(f'Processing {task} validation set...')
            # multi-processing
            with futures.ThreadPoolExecutor(NUM_WORKERS) as executor:
                executor.map(video2frames, val_task2path[task])


for subset in ['training', 'readout', 'test']:
    process_videos(subset)
