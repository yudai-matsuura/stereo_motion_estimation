from ultralytics import YOLO
import cv2

image_path = "data/000000_11.png"

img = cv2.imread(image_path)

model = YOLO("yolo26n.pt")

results = model(img)
result_img = results[0].plot()

# process per box
for box in results[0].boxes:
    class_id = int(box.cls)
    name = results[0].names[class_id]
    print(name)

cv2.imwrite(
    "outputs/kitti_detection.png",
    result_img
)

print("saved")