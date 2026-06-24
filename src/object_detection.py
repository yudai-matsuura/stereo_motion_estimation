import cv2

image_path = "/data/000000_11.png"

img = cv2.imread(image_path)

print(type(img))
print(img.shape)