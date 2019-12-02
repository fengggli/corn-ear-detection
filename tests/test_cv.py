import cv2
import numpy as np
video_path= '/share/Competition2/cornvideos/2inch/GOPR0165.MP4'
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
