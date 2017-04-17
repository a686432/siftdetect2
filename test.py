import cv2
import method
import numpy as np

weigh=np.load("a.npy")
img=cv2.imread("1.jpg")
sift = cv2.xfeatures2d.SIFT_create(100)
kp, res = sift.detectAndCompute(img, None)
img2, M = method.dataIncreasing_camera(img)
kp2, res2 = sift.detectAndCompute(img, None)
responses = np.array([k for k in range(0, len(kp))])
knn = cv2.ml.KNearest_create()
knn.train(res, cv2.ml.ROW_SAMPLE, responses)
print weigh
