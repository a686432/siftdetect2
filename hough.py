import cv2
import numpy as np

img = cv2.imread('2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,150,200,apertureSize = 3)
out = cv2.resize(edges,(1600,1600))
cv2.imshow("sift picture of translation", out)
minLineLength = 600
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,400,minLineLength,maxLineGap)
print len(lines)
for i in range(0,len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.waitKey(0)
cv2.imwrite('houghlines5.jpg',img)