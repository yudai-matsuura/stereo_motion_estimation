from ultralytics import YOLO
import cv2

left_color = cv2.imread(
    "data/left/000000_10.png"
)

left_gray = cv2.imread(
    "data/left/000000_10.png",
    cv2.IMREAD_GRAYSCALE
)

right_gray = cv2.imread(
    "data/right/000000_10.png",
    cv2.IMREAD_GRAYSCALE
)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=5
)

disparity = stereo.compute(
    left_gray,
    right_gray
)

model = YOLO("yolo26n.pt")

results = model(left_color)

result_img = results[0].plot()

for box in results[0].boxes:

    class_id = int(box.cls)
    name = results[0].names[class_id]

    if name != "car":
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    disp = disparity[cy, cx] / 16.0

    print(
        f"car center=({cx},{cy}) "
        f"disparity={disp:.2f}"
    )

    cv2.circle(
        result_img,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    "outputs/car_disparity.png",
    result_img
)

print("saved")