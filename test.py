import cv2
import method
import numpy as np


def inputimage():
    basefilename = "res/40108/"
    inputfilename = "data/40108/"
    img = cv2.imread(inputfilename + "0.jpg")
    img2, M = method.dataIncreasing_camera(img)
    #img2 = cv2.imread(inputfilename + "1.jpg")
    img2 = cv2.resize(img2, (1000, 1000))
    weigh = np.load(basefilename + 'a' + str(nfeature) + '.npy')
    distinguish = np.load(basefilename + 'c' + str(nfeature) + '.npy')
    appearance = np.load(basefilename + 'd' + str(nfeature) + '.npy')
    img = cv2.resize(img, (1000, 1000))
    return weigh, distinguish, appearance, img, img2


def siftKnn():
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp, res = sift.detectAndCompute(img, None)
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp2, res2 = sift.detectAndCompute(img2, None)
    return kp, res,kp2, res2



if __name__ == "__main__":
    MIN_MATCH_COUNT = 5
    nfeature = 1000
    npoint = 100

    weigh, distinguish, appearance, img, img2 = inputimage()
    kp, res, kp2, res2 = siftKnn()
    matchs = []
    appearance_sort = sorted(appearance, reverse=True)
    res0 = []
    responses = []
    transform = []
    th = appearance_sort[npoint]
    th=0
    lists = []
    out = img
    knn = cv2.ml.KNearest_create()


    m = np.zeros(len(kp))
    # for i in range(0, len(kp)):
    #     k = 0
    #     if appearance[i] >= th:
    #         out = cv2.circle(out, (int(kp[i].pt[0]), int(kp[i].pt[1])), 2, (55, 255, 155), 2)
    #         res0.append(res[i])
    #         responses.append(i)
    #         m[i] = k
    #         k += 1
    #         a = []
    #         lists.append(a)
    # responses = np.array(responses)
    # res0 = np.array(res0)
    # print responses
    # cv2.imshow("strong sift descriptor", out)
    knn.train(res, cv2.ml.ROW_SAMPLE, responses)
    out = img
    ret, results, neighbours, dist = knn.findNearest(res2, 3)
    print neighbours
    cv2.waitKey(0)
    print len(neighbours)

    for i in range(0, len(neighbours)):
        #print "Match:" + str(dist[i]) + "Weigh:" + str(weigh[int(neighbours[i][0])]) + "Num" + str(
         #   int(neighbours[i][0]))
        if dist[i][0] < min(weigh[int(neighbours[i][0]), 1] , weigh[
            int(neighbours[i][0]), 0]):  # and dist[i][0]/dist[i][1]<distinguish[1,int(neighbours[i][0])]:
            match = cv2.DMatch(int(neighbours[i][0]), i, 3)
            print "Good Match"
            matchs.append(match)
            # lists[]
            print int(neighbours[i][0]),i
    if len(matchs) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp[m.queryIdx].pt for m in matchs[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matchs[:50]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if len(img.shape) > 2:
            h, w, d = img.shape
            pts = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(matchs), MIN_MATCH_COUNT)
        matchesMask = None

    out = cv2.drawMatches(img, kp, img2, kp2, matchs, out)
    out = cv2.resize(out, (1600, 1000))
    cv2.imshow("sift picture of translation", out)
    cv2.imshow("sift picture of translation2", img2)
    cv2.waitKey(0)
    print weigh
