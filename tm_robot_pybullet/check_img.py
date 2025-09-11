import cv2 as cv

img_path = "rlds_dataset/episode_001/img/img_0000.png"
img = cv.imread(img_path)
height, width, channel = img.shape
print(height, width, channel)