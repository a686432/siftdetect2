# coding=utf-8
import cv2
import numpy as np


def dataIncreasing_camera(img):
    #这里包含了相机的视角的变换
    ax = np.random.uniform(0,np.pi/4)
    ay = np.random.uniform(0,np.pi/4)
    az = np.random.uniform(0,np.pi/6)
    gAlpha=np.random.uniform(15,100)
    gGamma=2
    s = np.random.uniform(0.5,2)
    height, width, channel = img.shape
    pts = [[-height / 2, -width / 2, 0], [-height / 2, width / 2, 0], [height / 2, -width / 2, 0],
           [height / 2, width / 2, 0]]
    pts = np.array([pts])
    ptsv = cv2.convertPointsToHomogeneous(pts)
    Rs=np.mat([[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]])
    Rx = np.mat([[1, 0, 0, 0], [0, np.cos(ax), np.sin(ax), 0], [0, -np.sin(ax), np.cos(ax), 0], [0, 0, 0, 1]])
    Ry = np.mat([[np.cos(ay), 0, np.sin(ay), 0], [0, 1, 0, 0], [-np.sin(ay), 0, np.cos(ay), 0], [0, 0, 0, 1]])
    Rz = np.mat([[np.cos(az), np.sin(az), 0, 0], [-np.sin(az), np.cos(az), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    dspts = (Rz * Ry * Rx *Rs *ptsv.T).T
    dst = dspts[:, 0:3]
    d = np.array(
        [[x[0, 0] * height / (height + x[0, 2]), x[0, 1] * width / (x[0, 2] + width)] for x in
         dst])
    H, k = cv2.findHomography(pts, d)
    # out = cv2.warpPerspective(img, H, (height * 2, width * 2))
    # l=np.float32(1)
    #
    # cv2.pow(out,l,out)
    out=img
    cv2.addWeighted(out, (gAlpha / 50.0), out, 0, gGamma, out)
    return out, H



