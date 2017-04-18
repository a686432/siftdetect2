import cv2
import numpy as np
import method


def siftComp(img1, nloop):
    sift = cv2.xfeatures2d.SIFT_create(nfeature)
    kp, res = sift.detectAndCompute(img1, None)
    pts = np.array([[[k.pt[0], k.pt[1]] for k in kp]])
    responses = np.array([k for k in range(0, len(kp))])
    knn = cv2.ml.KNearest_create()
    knn.train(res, cv2.ml.ROW_SAMPLE, responses)
    sift = cv2.xfeatures2d.SIFT_create(1000)
    for i in range(0, nloop):
        print "******This is the " + str(i) + " loop*******"
        img2, M = method.dataIncreasing_camera(img1)
        dst = cv2.perspectiveTransform(pts, M)
        dst = [(int(k[0]), int(k[1])) for k in dst[0]]
        kp2, res2 = sift.detectAndCompute(img2, None)
        for k in dst:
            img2 = cv2.circle(img2, k, 2, (55, 255, 155), 2)
        out = cv2.drawKeypoints(img2, kp2, img2)
        matchs = []
        nTure = []
        nFalse = []
        nnTure = []
        nnFalse = []
        ret, results, neighbours, dist = knn.findNearest(res2, 3)
        for k in range(0, len(neighbours)):
            # print dst[int(neighbours[k][0])]

            (x0, y0) = dst[int(neighbours[k][0])]
            (x1, y1) = kp2[k].pt
            err = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            if err < 10:
                img2 = cv2.circle(img2, (x0, y0), 6, (255, 155, 55), 1)
                print "GOOD NO." + str(int(neighbours[k][0])) + "\nthe shortest distence is:" + str(dist[k])
                if dist[k][0] < predict[int(neighbours[k][0])]:
                    print "Preduct is True"
                    nTure.append(int(neighbours[k][0]))
                else:
                    print "Preduct is False"
                    nFalse.append(int(neighbours[k][0]))
                if dist[k][0] > weigh[int(neighbours[k][0]), 0]:
                    weigh[int(neighbours[k][0]), 0] = dist[k][0]
                    predict[int(neighbours[k][0])] = min(weigh[int(neighbours[k][0]), 0],
                                                         weigh[int(neighbours[k][0]), 1])
                    match = cv2.DMatch(int(neighbours[k][0]), k, 3)
                    matchs.append(match)
            else:
                print "BAD NO." + str(int(neighbours[k][0])) + "\nthe shortest distence is:" + str(dist[k])
                if dist[k][0] < predict[int(neighbours[k][0])]:
                    print "Preduct is False"
                    nnFalse.append(int(neighbours[k][0]))
                else:
                    print "Preduct is True"
                    nnTure.append(int(neighbours[k][0]))
                if 0 == weigh[int(neighbours[k][0]), 1] or dist[k][0] < weigh[int(neighbours[k][0]), 1]:
                    weigh[int(neighbours[k][0]), 1] = dist[k][0]
                    predict[int(neighbours[k][0])] = min(weigh[int(neighbours[k][0]), 0],
                                                         weigh[int(neighbours[k][0]), 1])
        print "The number of true match and false match is " + str((len(nTure), len(nFalse), len(nTure)*1.0 / (len(nFalse)+len(nTure))))
        print "The number of true match and false match is " + str((len(nTure), len(nnFalse), len(nTure)*1.0 / (len(nnFalse)+len(nTure))))

    out = cv2.drawMatches(img1, kp, img2, kp2, matchs, out)
    cv2.imshow("sift picture of translation", out)
    return


nfeature = 500
#weigh = np.zeros((nfeature, 2))
weigh=np.load("a.npy")
predict=np.load("b.npy")
#predict = np.zeros(nfeature)
img = cv2.imread("2.jpg")
img1 = cv2.imread("2.jpg")
img = cv2.resize(img, (1000, 1000))
img1 = cv2.resize(img1, (1000, 1000))
sift = cv2.xfeatures2d.SIFT_create(nfeature)
kp, res = sift.detectAndCompute(img1, None)
out = cv2.drawKeypoints(img1, kp, img1)
# cv2.imshow("sift picture of orginal", out)
siftComp(img, 100)
print weigh
np.save("a", weigh)
np.save("b", predict)
cv2.waitKey(0)
