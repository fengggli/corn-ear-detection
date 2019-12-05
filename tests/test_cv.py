import cv2
import numpy as np

# video are all muted, using command: 
# for file in ./2inch/*; do ffmpeg -i $file -vcodec copy -an 2inch_muted/`basename $file`; done
# see https://stackoverflow.com/questions/49060054/opencv-videocapture-closes-with-videos-from-gopro
video_path= '/share/Competition2/cornvideos/2inch_muted/GOPRO165.MP4'
#video_path= '/share/Competition2/cornvideos/4inch/GOPR0388.MP4'# 4 inch is fine
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture("test.avi") # it's also not working
if not cap.isOpened(): # Output: False
    print("Open error")

ret, image_np = cap.read()
if not ret:
    print("Read error")
else:
    print("Fine, contents:", image_np)
