import cv2
import method
import numpy as np

weigh=np.load("a1000.npy")
distinguish=np.load("c1000.npy")
img=cv2.imread("2.jpg")
appearance=np.load("d1000.npy")
img=cv2.resize(img,(1000,1000))
matchs=[]
sift = cv2.xfeatures2d.SIFT_create(1000)
kp, res = sift.detectAndCompute(img, None)
appearance_sort=sorted(appearance,reverse=True)
res0=[]
responses=[]
th=appearance_sort[100]
out=img
for i in range(0,len(kp)):
    if appearance[i]>=th:
        out = cv2.circle(out, (int(kp[i].pt[0]),int(kp[i].pt[1])), 2, (55, 255, 155), 2)
        res0.append(res[i])
        responses.append(i)
responses=np.array(responses)
res0=np.array(res0)
print responses
cv2.imshow("strong sift descriptor", out)
cv2.waitKey(0)

sift = cv2.xfeatures2d.SIFT_create(1000)
img2, M = method.dataIncreasing_camera(img)
img2=cv2.imread("data/40103/1.jpg")
img2=cv2.resize(img2,(1000,1000))
kp2, res2 = sift.detectAndCompute(img2, None)

# responses = np.array([k for k in range(0, len(kp))])
# print responses

knn = cv2.ml.KNearest_create()
knn.train(res0, cv2.ml.ROW_SAMPLE, responses)
out=img
# for i in range(0,len(kp)):
#     out = cv2.circle(img, (int(kp[i].pt[0]),int(kp[i].pt[1])), int(20*distinguish[0,i]), (255, 255, int(255*distinguish[1,i])), 1)
#     out = cv2.circle(img, (int(kp[i].pt[0]), int(kp[i].pt[1])), int(20 * distinguish[1, i]),
#                      (255, 0, 255), 1)
ret, results, neighbours, dist = knn.findNearest(res2, 3)
for i in range(0, len(neighbours)):
    print "Match:"+str(dist[i])+"Weigh:"+str(weigh[int(neighbours[i][0])])+"Num"+str(int(neighbours[i][0]))
    if weigh[int(neighbours[i][0]),0]<2000000:
        if dist[i][0]<(weigh[int(neighbours[i][0]), 0]+weigh[int(neighbours[i][0]), 1])/2 and distinguish[0,int(neighbours[i][0])]>0:
            match = cv2.DMatch(int(neighbours[i][0]), i, 3)
            print "Good Match"
            matchs.append(match)
out = cv2.drawMatches(img, kp, img2, kp2, matchs,out)
out = cv2.resize(out,(1600,1000))
cv2.imshow("sift picture of translation", out)
cv2.waitKey(0)
print weigh
