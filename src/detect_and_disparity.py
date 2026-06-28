from ultralytics import YOLO
import numpy as np
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

# stereo matching
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=9
)
disparity = stereo.compute(
    left_gray,
    right_gray
)

# convert disparity image
disp_vis = cv2.normalize(
    disparity,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

disp_vis = disp_vis.astype("uint8")

cv2.imwrite(
    "outputs/disparity.png",
    disp_vis
)

# detection
model = YOLO("yolo26n.pt")
results = model(left_color)
result_image = results[0].plot()

for box in results[0].boxes:
    class_id = int(box.cls)
    name = results[0].names[class_id]

    if name != "car":
        continue

    confidence = float(box.conf)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # disp = disparity[cy, cx] / 16.0

    # Use the median value within the BBOX
    roi = disparity[y1:y2, x1:x2] / 16.0
    valid_disp = roi[roi > 0]
    disp = np.median(valid_disp)

    if disp > 0:
        depth = (721.5377 * 0.54) / disp
    else:
        disp = -1

    print(
        f"confidence={confidence:.3f}"
        f"car center=({cx},{cy}) "
        f"disparity={disp:.2f}"
        f"depth={depth:.2f}m"
    )

    cv2.putText(
        result_image,
        f"{depth:.1f}m",
        (cx + 10, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

    cv2.circle(
        result_image,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    "outputs/car_disparity.png",
    result_image
)

print("saved")
