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
    # print(name)

    if name != "car":
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # print(x1, y1, x2, y2)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    print(cx, cy)

    #bbox
    # cv2.rectangle(
    #     img,
    #     (x1, y1),
    #     (x2, y2),
    #     (0, 255, 0),
    #     2
    # )

    # center
    cv2.circle(
        result_img,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    "outputs/kitti_detection.png",
    result_img
)

print("saved")