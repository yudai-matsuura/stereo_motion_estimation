import cv2

image_path_left = "data/left/000000_10.png"
image_path_right = "data/right/000000_10.png"

left_img = cv2.imread(image_path_left, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(image_path_right, cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=5
)

disparity = stereo.compute(left_img, right_img)

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
