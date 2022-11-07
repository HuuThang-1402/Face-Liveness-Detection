# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import pandas as pd
from glob import glob
import os
import argparse
import cv2
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

video_dirs = 'public/videos'
videos = glob(os.path.join(video_dirs, '*'))

video_names = []
predictions = []

count_video = 1

for video in videos:
    count=0
    video_name = os.path.basename(video)
    video_names.append(video_name)
    prediction = []
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, 4800)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=600)
            # grab the frame dimensions and convert it to a blob
            face = cv2.resize(frame, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(face)[0][1]
            prediction.append(preds)

        confidence_score = sum(prediction)/len(prediction)

    except:
        confidence_score = 0
    
    print(count_video, video_name, confidence_score)
    predictions.append(confidence_score)
    count_video+=1

df = pd.DataFrame(columns=['fname', 'liveness_score'])
df['fname'] = video_names
df['liveness_score'] = predictions
df.to_csv("predict.csv", index = False)
        