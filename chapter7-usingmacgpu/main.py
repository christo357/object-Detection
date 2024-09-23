import cv2
import numpy as np
from ultralytics import YOLO

# import torch
# print(torch.backends.mps.is_available())

cap = cv2.VideoCapture('../Videos/people.mp4')

model = YOLO('../yolo-weights/yolov8m.pt')

while True:
    success, frame = cap.read()
    if not success:
        break
    
    results = model(frame, device='mps')
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')
    for cls,bbox in zip(classes, bboxes):
        (x1,y1, x2, y2) = bbox
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255),2)
        cv2.putText(frame, str(cls), (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0, 255), 2)
    
    cv2.imshow('Img', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()                    # releasing the captured object holding the video
cv2.destroyAllWindows()          # close all opencv windows that might be opened