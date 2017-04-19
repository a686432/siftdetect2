import cv2
import method
import numpy as np

weigh=np.load("a.npy")
img=cv2.imread("2.jpg")
print weigh
img=cv2.resize(img,(1000,1000))
matchs=[]
sift = cv2.xfeatures2d.SIFT_create(500)
kp, res = sift.detectAndCompute(img, None)
sift = cv2.xfeatures2d.SIFT_create(1000)
img2, M = method.dataIncreasing_camera(img)
img2=cv2.imread("3.jpg")
img2=cv2.resize(img2,(700,700))
kp2, res2 = sift.detectAndCompute(img2, None)
responses = np.array([k for k in range(0, len(kp))])
knn = cv2.ml.KNearest_create()
knn.train(res, cv2.ml.ROW_SAMPLE, responses)
out = cv2.drawKeypoints(img2, kp2, img2)
ret, results, neighbours, dist = knn.findNearest(res2, 3)
for i in range(0, len(neighbours)):
    print "Match:"+str(dist[i])+"Weigh:"+str(weigh[int(neighbours[i][0])])+"Num"+str(int(neighbours[i][0]))
    if dist[i][0]<min(weigh[int(neighbours[i][0]), 1],weigh[int(neighbours[i][0]), 0]):
        match = cv2.DMatch(int(neighbours[i][0]), i, 3)
        print "Good Match"
        matchs.append(match)
out = cv2.drawMatches(img, kp, img2, kp2, matchs,out)
out = cv2.resize(out,(1600,1600))
cv2.imshow("sift picture of translation", out)
cv2.waitKey(0)
print weigh
