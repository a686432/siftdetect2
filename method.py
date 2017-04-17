# coding=utf-8
import cv2
import numpy as np


def dataIncreasing_camera(img):
    # 这里包含了相机的视角的变换
    ax = np.random.uniform(0,np.pi/4)
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
        [[x[0, 0] * height / (height + x[0, 2]), x[0, 1] * width / (x[0, 2] + width)] for x in
         dst])
    H, k = cv2.findHomography(pts, d)
    out = cv2.warpPerspective(img, H, (height * 3, width * 3))
    return out, H


def siftComp(img1, nloop):
    sift = cv2.xfeatures2d.SIFT_create(100)
    kp, res = sift.detectAndCompute(img1, None)
    pts = np.array([[[k.pt[0], k.pt[1]] for k in kp]])
    responses = np.array([k for k in range(0, len(kp))])
    knn = cv2.ml.KNearest_create()
    knn.train(res, cv2.ml.ROW_SAMPLE, responses)

    for i in range(0, nloop):
        img2, M = dataIncreasing_camera(img1)
        dst = cv2.perspectiveTransform(pts, M)

        dst = [(int(k[0]), int(k[1])) for k in dst[0]]
        print dst[0]
        kp2, res2 = sift.detectAndCompute(img2, None)
        for k in dst:
            img2 = cv2.circle(img2, k, 2, (55, 255, 155), 2)
        out = cv2.drawKeypoints(img2, kp2, img2)
        matchs = []
        #cv2.waitKey(0)
        ret, results, neighbours, dist = knn.findNearest(res2, 3)
        print (neighbours[0], dist[0])
        print dst[1]
        for k in range(0, len(neighbours)):
            # print dst[int(neighbours[k][0])]
            (x0, y0) = dst[int(neighbours[k][0])]
            (x1, y1) = kp2[k].pt
            err = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            if err < 4:
                img2 = cv2.circle(img2, (x0, y0), 6, (255, 155, 55), 1)
                print "GOOD NO." + str(int(neighbours[k][0])) + "\nthe shortest distence is:" + str(dist[k])
                if dist[k][0]>weigh[int(neighbours[k][0]),0]:
                    weigh[int(neighbours[k][0]), 0]=dist[k][0]
                    match = cv2.DMatch(int(neighbours[k][0]), k, 3)
                    matchs.append(match)
            else:
                print "BAD NO." + str(int(neighbours[k][0])) + "\nthe shortest distence is:" + str(dist[k])
                if 0==weigh[int(neighbours[k][0]), 1] or dist[k][0] < weigh[int(neighbours[k][0]), 1]:
                    weigh[int(neighbours[k][0]), 1] = dist[k][0]


        out=cv2.drawMatches(img1,kp,img2,kp2,matchs,out)
        cv2.imshow("sift picture of translation", out)


    return
nfeature=100
weigh=np.zeros((nfeature,2))

img = cv2.imread("1.jpg")
img1 = cv2.imread("1.jpg")
sift = cv2.xfeatures2d.SIFT_create(100)
kp, res = sift.detectAndCompute(img1, None)
out = cv2.drawKeypoints(img1, kp, img1)
#cv2.imshow("sift picture of orginal", out)
siftComp(img, 100)
print weigh
np.save("a",weigh)
cv2.waitKey(0)
