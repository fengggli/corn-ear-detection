import cv2
video_path='./GOPR0165.MP4'
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture("test.avi") # it's also not working
if not cap.isOpened(): # Output: False
    print("Open error")

ret = cap.read()
if not ret:
    print("Read error")
else:
    print("Fine")
