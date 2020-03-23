# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:52:24 2018

@author: hamdd
"""
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from .pyramid import opticalFlowPyr
from .utils import matlab_style_gauss2D

def opticalFlowLK(I1, I2, wsize=10, **kwargs):
#function [Vx,Vy,reliab] = opticalFlowLk( I1, I2, radius  )
#% Compute elements of A'A and also of A'b
    wsize=np.min([wsize,int(np.floor(np.min([I1.shape[0],I1.shape[1]])/2))-1])
    
#    Ix,Iy=gradient2(I1)
    d = np.multiply([-1,8,0,-8,1],1/12.);
    Ix = convolve2d(I1,             [d] , mode='same');
    Iy = convolve2d(I1,np.transpose([d]), mode='same');
    
    It=I2-I1
    
#    h = matlab_style_gauss2D([3,3], wsize);
#    I1 = convolve(I1,h, mode='nearest');
#    I2 = convolve(I2,h, mode='nearest');
    
    win = np.array([np.ones(wsize)])
    AAxy=convolve2d(np.multiply(Ix, Iy),win, mode='same')
    
    AAxx=convolve2d(   np.power(Ix,  2),win, mode='same')+1e-5
    AAyy=convolve2d(   np.power(Iy,  2),np.transpose(win), mode='same')+1e-5
                    
    ABxt=convolve2d(np.multiply(-Ix,It),win, mode='same')
    AByt=convolve2d(np.multiply(-Iy,It),np.transpose(win), mode='same')
#    % Find determinant and trace of A'A
    AAdet=np.subtract(np.multiply(AAxx,AAyy),np.power(AAxy,2))
    AAdeti=np.divide(1,AAdet);
    AAdeti[AAdeti == np.inf] = 0
    AAtr=AAxx+AAyy
#    % Compute components of velocity vectors (A'A)^-1 * A'b
    Vx = np.multiply(AAdeti, np.subtract(np.multiply( AAyy,ABxt), np.multiply(AAxy,AByt)));
    Vy = np.multiply(AAdeti,      np.add(np.multiply(-AAxy,ABxt), np.multiply(AAxx,AByt)));
#    % Check for ill conditioned second moment matrices
    reliab = np.multiply(0.5,AAtr) - np.multiply(0.5,np.sqrt(np.power(AAtr,2)-np.multiply(4,AAdet)));
    return Vx, Vy, reliab

def opticalFlowLKPyr(I1, I2, **kwargs):
    Vx, Vy, __ = opticalFlowPyr(I1, I2, opticalFlowLK, \
                                **kwargs)
    return np.stack((Vx, Vy), axis=2)
