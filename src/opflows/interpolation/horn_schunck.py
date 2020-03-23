# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:07:43 2018

Code inspired by: https://github.com/scivision/pyoptflow/blob/master/pyoptflow/hornschunck.py

@author: hamdd
"""
from .pyramid import opticalFlowPyr
from scipy.ndimage.filters import convolve as filter2
from scipy.signal import convolve2d
import numpy as np

#import matplotlib.pyplot as plt
#import cv2

#
HSKERN =np.array([[1/12, 1/6, 1/12],
                  [1/6,    0, 1/6],
                  [1/12, 1/6, 1/12]],float)

kernelX = np.array([[-1, 1],
                     [-1, 1]]) * .25 #kernel for computing d/dx

kernelY = np.array([[-1,-1],
                     [ 1, 1]]) * .25 #kernel for computing d/dy

kernelT = np.ones((2,2))*.25


def computeDerivatives(im1, im2):

    fx = filter2(im1,kernelX) + filter2(im2,kernelX)
    fy = filter2(im1,kernelY) + filter2(im2,kernelY)

   # ft = im2 - im1
    ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

    return fx,fy,ft

def opticalFlowHS(I1, I2, alpha=1, nIter=250, **kwargs):
    """
    Computes the Optical Flow using Horn-Schunck method.
    This code is inspired by a Matlab Code found here:
        https://github.com/pdollar/toolbox/blob/master/videos/opticalFlow.m
    
    """
#    pad = lambda I, p: np.pad(I, p, 'reflect')
    crop = lambda I, c: I[c:(-1-c), c:(-1-c)]
#    gaus = lambda factor, matrix : np.multiply(factor, np.array(matrix))
#    Ex = convolve2d(I1, gaus(0.25,[[-1, 1],[ -1, 1]]),'same') + convolve2d(I2, gaus(0.25,[[-1, 1],[ -1, 1]]),'same');
#    Ey = convolve2d(I1, gaus(0.25,[[-1, -1],[ 1, 1]]),'same') + convolve2d(I2, gaus(0.25,[[-1, -1],[ 1, 1]]),'same');
#    Et = convolve2d(I1, np.array([np.multiply(0.25,np.ones(2))]) ,'same') + convolve2d(I2, np.array([np.multiply(-0.25,np.ones(2))]),'same');
    Ex, Ey, Et = computeDerivatives(I1, I2)
    Z=np.divide(1,(alpha*alpha + Ex*Ex + Ey*Ey)); 
    reliab=crop(Z,1);
    #% iterate updating Ux and Vx in each iter
    if( 0 ):
        pass
#      [Vx,Vy]=opticalFlowHsMex(Ex,Ey,Et,Z,nIter);
    #  Vx=crop(Vx,1); Vy=crop(Vy,1);
    else:
        Vx=np.zeros(I1.shape,'single'); 
        Vy=Vx.copy();
        f=np.single([[0, 1, 0],[ 1, 0, 1],[ 0, 1, 0]])/4;
        for i in range(0,nIter):
            Mx=convolve2d(Vx,f,'same'); 
            My=convolve2d(Vy,f,'same');
            m=np.multiply((np.multiply(Ex,Mx)+np.multiply(Ey,My)+Et),Z); 
            Vx=Mx-np.multiply(Ex,m); 
            Vy=My-np.multiply(Ey,m);
    return Vx, Vy, reliab

def opticalFlowHSPyr(I1, I2, **kwargs):
    Vx, Vy, __ = opticalFlowPyr(I1, I2, opticalFlowHS, **kwargs)
    return np.stack((Vx, Vy), axis=2)