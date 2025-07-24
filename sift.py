import cv2 as cv
import matplotlib.pyplot as plt

img_path = 'images/Road.png'

img_gray = cv.imread(img_path,cv.IMREAD_GRAYSCALE)


sift = cv.SIFT_create()
keypoints = sift.detectAndCompute(img_gray,None)
img_with_keypoints = cv.drawKeypoints(img_gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


plt.figure()
plt.imshow(img_with_keypoints)
plt.show()
