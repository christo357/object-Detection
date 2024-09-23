from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8l.pt')
results = model('Images/img3.jpg', show=True)

cv2.waitKey(10000)