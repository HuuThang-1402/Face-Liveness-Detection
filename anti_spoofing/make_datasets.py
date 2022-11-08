import cv2
import os
from pathlib import Path
from tqdm import tqdm
from glob import glob
import argparse
import pandas as pd
import shutil

def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def make_data(args):
    label = pd.read_csv(os.path.join(args.root, 'label.csv'))
    list_video = glob(os.path.join(args.root, 'videos', '*'))
    
    cls_fake_path = os.path.join(args.dest, 'spoof')
    cls_real_path = os.path.join(args.dest, 'real')

    make_if_not_exist(cls_fake_path)
    make_if_not_exist(cls_real_path)

    for video in tqdm(list_video):
        video_name = os.path.basename(video)
        cls = label[label['fname'] == video_name]['liveness_score'].values[0]
        if cls == 0:
            shutil.copy(video_name, cls_fake_path)
        else:
            shutil.copy(video_name, cls_real_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='train')
    parser.add_argument('--dest', type=str, default='datasets')

    args = parser.parse_args()
    make_data(args)