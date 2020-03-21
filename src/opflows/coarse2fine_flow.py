import numpy as np
import pyflow
import os
import cv2
import time
import visualization

im1 = cv2.imread("../datasets/of_pred/000045_10.png")
im2 = cv2.imread("../datasets/of_pred/000045_11.png")
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30

def coarse2fine_flow(im1,im2):
    im1_f = im1.astype("float") / 255.0
    im2_f = im2.astype("float") / 255.0
    
    #check if image is grayscale
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    if(im1_f.ndim < 3):
        im1_f = im1_f[:, :, np.newaxis]
        im2_f = im2_f[:, :, np.newaxis]
        colType = 1
    elif(im1_f.shape[2] < 3):
        colType = 1
        
    u, v, im2W = pyflow.coarse2fine_flow( im1_f, im2_f, alpha, ratio, minWidth,
                                         nOuterFPIterations, nInnerFPIterations,
                                         nSORIterations, colType)

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    im_warped = (im2W[:, :, ::-1] * 255).astype("uint8")
    
    return flow, im_warped


if __name__ == "__main__":
    image1_path = "../datasets/of_pred/000045_10.png"
    image2_path = "../datasets/of_pred/000045_11.png"
    
    im1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    flow, im_warped = coarse2fine_flow(im1,im2)

    rgb_flow1 = visualization.colorflow_black(flow)
    rgb_flow2 = visualization.colorflow_white(flow)
    
    cv2.imshow("nuevo",rgb_flow1)
    cv2.imshow("nuevo2",rgb_flow2)
    cv2.imshow("im_warped",im_warped)
    cv2.waitKey()