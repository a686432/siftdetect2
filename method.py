# coding=utf-8
import cv2
import numpy as np


def dataIncreasing_camera(img):
    # 这里包含了相机的视角的变换
    ax = 0.1
    ay = 0
    az = 0
    height, width, channel = img.shape
    pts = [[-height / 2, -width / 2, 0], [-height / 2, width / 2, 0], [height / 2, -width / 2, 0],
           [height / 2, width / 2, 0]]
    pts = np.array([pts])
    ptsv = cv2.convertPointsToHomogeneous(pts)
    Rx = np.mat([[1, 0, 0, 0], [0, np.cos(ax), np.sin(ax), 0], [0, -np.sin(ax), np.cos(ax), 0], [0, 0, 0, 1]])
    Ry = np.mat([[np.cos(ay), 0, np.sin(ay), 0], [0, 1, 0, 0], [-np.sin(ay), 0, np.cos(ay), 0], [0, 0, 0, 1]])
    Rz = np.mat([[np.cos(az), np.sin(az), 0, 0], [-np.sin(az), np.cos(az), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    dspts = (Rz * Ry * Rx * ptsv.T).T
    dst = dspts[:, 0:3]
    d = np.array(
        [[x[0, 0] * height / (height + x[0, 2]) + height, x[0, 1] * width / (x[0, 2] + width) + width / 2] for x in
         dst])
    H, k = cv2.findHomography(pts, d)
    out = cv2.warpPerspective(img, H, (height * 3, width * 3))
    return out


def siftComp(img1, nloop):
    sift = cv2.xfeatures2d.SIFT_create(100)
    kp, res = sift.detectAndCompute(img1, None)
    for i in range(0, nloop):
        img2 = dataIncreasing_camera(img1)
        knn.train(traindata, responses)
        kp2, res2 = sift.detectAndCompute(img2, None)
        for j in range(0,len(kp)):
            (x,y)=kp[j].pt
            knn = cv2.KNearest()

            ret, results, neighbours, dist = knn.find_nearest(newcomer, 3)
            print (x,y)
    return


img = cv2.imread("1.jpg")
siftComp(img,100)
cv2.imshow("1", out)

cv2.waitKey(0)
