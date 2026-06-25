from ultralytics import YOLO
import cv2

image_path = "data/000000_11.png"

img = cv2.imread(image_path)

model = YOLO("yolo26n.pt")

results = model(img)
result_img = results[0].plot()
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
    print(x1, y1, x2, y2)

cv2.imwrite(
    "outputs/kitti_detection.png",
    result_img
)

print("saved")