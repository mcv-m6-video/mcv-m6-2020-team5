# -*- coding: utf-8 -*-
"""
Created on Thu May 24 00:07:17 2018

@author: hamdd
"""
import cv2
import numpy as np
#from .utils import get_gray, of2rgb,warp_flow

from .pyramid import opticalFlowPyr

def tvl1_simple(img1, img2, imtype="none", Lambda=0.9, **kwargs):
#    gray1 = get_gray(img1,imtype)
#    gray2 = get_gray(img2,imtype)
    tvl1 = cv2.createOptFlow_DualTVL1()
    
#    opflow = np.zeros_like(gray1)
    tvl1.setLambda(Lambda)
    
    print("Running...", img1.shape)
    img1 = np.array(img1, dtype = np.float32)
    img2 = np.array(img2, dtype = np.float32)
    opflow = tvl1.calc(img1, img2, None)
    
#    rgb = of2rgb(opflow[:,:,0],opflow[:,:,1])
#    imgi = warp_flow(img1, opflow, 2.0)
    
#    return imgi, rgb
    return opflow[:,:,0], opflow[:,:,1], np.nan

def opticalFlowTVL1Pyr(I1, I2, **kwargs):
    Vx, Vy, __ = opticalFlowPyr(I1, I2, tvl1_simple, \
                                **kwargs)
    return np.stack((Vx, Vy), axis=2)