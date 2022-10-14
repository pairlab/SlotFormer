import os
from concurrent import futures as futures

from nerv.utils import glob_all, VideoReader


def video2frames(video_path):
    cap = VideoReader(video_path)
    frame_dir = video_path[:-4]  # '.mp4'
    if os.path.exists(frame_dir):  # already exists
        return
    cap.cvt2frames(frame_dir, target_shape=RESIZE)


data_root = './data/CLEVRER/'
splits = ['train', 'val', 'test']
RESIZE = (128, 128)
NUM_WORKERS = 8

for split in splits:
    video_dir = os.path.join(data_root, 'videos', split)
    video_subdirs = glob_all(video_dir, only_dir=True)
    video_files = []
    for subdir in video_subdirs:
        video_files += glob_all(subdir)
    print(f'Converting {split} set {len(video_files)} videos...')
    # multi-processing
    with futures.ThreadPoolExecutor(NUM_WORKERS) as executor:
        executor.map(video2frames, video_files)
