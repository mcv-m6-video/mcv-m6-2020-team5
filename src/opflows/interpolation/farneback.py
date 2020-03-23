# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:52:04 2018

@author: hamdd
"""

import cv2
import numpy as np


def calc_opt_flow(f1, f2):
    
    
#    rgb = of2rgb(flow[...,0], flow[...,1])
    
    return flow

def farneback(img1, img2, ftype="GRAY", imtype="rgb",minScale=1,wsize=15,\
              fIter=3,**kwargs):
    """
    img1: image at t0
    img2: image at t1
    ftype: type of farneback interpolation mode:
        GRAY: Uses same optical flow computated as if it were a simple layer 
            in grayscale mode. Converts RGB to GRAYSCALE and computes same
            optical flow to all 3 channels.
        LAYERED: Uses different optical flow to each layer of the image.
        
    """
    pyrLevels = int(np.sqrt(int(1/minScale)))
    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5,pyrLevels, wsize, \
                                        fIter, 5, 1.2, 0)
#    warped_img = warp_flow(img_to_warp, flow, -2.0)

    return flow


