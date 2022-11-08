import cv2
import os
from pathlib import Path
from tqdm import tqdm
from glob import glob
import argparse
import pandas as pd

def make_data(args):
    train_real = "datasets/train/real/"
    train_spoof = "datasets/train/spoof/"
    test_real = "datasets/test/real/"
    test_spoof = "datasets/test/spoof/"
    idx = 0
    for i in [train_real, train_spoof, test_real, test_spoof]:
        list_video = glob(i+"*")
        idx = 0
        for video in tqdm(list_video):
            idx = extract_frame(video, i, idx, args.skip)

        for vid in glob.glob(i+"*.mp4"):
            os.remove(vid)

        print(f"Number of video on folder {i}:",len(glob(i+"*mp4")))

def extract_frame(video_path, dest, idx, skip=7):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            file_path = os.path.join(dest, str(idx) + '.jpg')
            cv2.imwrite(file_path, frame)
            idx += 1

        count += 1

    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=7)

    args = parser.parse_args()
    make_data(args)