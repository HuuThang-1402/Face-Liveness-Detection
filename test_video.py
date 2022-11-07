from cupshelpers import missingExecutables
import cv2
import os
from glob import glob 

video = "/media/estella/OS/HocTap/Zalo_Challenge/public_test/public/videos/215.mp4"
video_name = os.path.basename(video)


cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_MSEC, 4800)
count = 0 
while True:
    ret, frame = cap.read()
    count+=1
    print(count)

    if not ret:
        break

cv2.destroyAllWindows()