# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 01:34:53 2018

@author: hamdd
"""
import numpy as np
import cv2
from scipy.ndimage import convolve
from .utils import matlab_style_gauss2D
from scipy.signal import medfilt2d

import matplotlib.pyplot as plt

from .utils import warp_flow2,warp_flow

dfs ={'smooth':1,
      'filt':5,
      'minScale':1/16,
      'maxScale':1,
      'windowSize':10,
      'nBlock':5,
      'alpha':1,
      'nIter':250
      };
      
def opticalFlowPyr(I1, I2, opflowFunction, smooth=1, minScale=1/16, \
                maxScale=1, nBlock=5, filt= 7, reversedPyr=False, **kwargs):
    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.imshow(I1)
#    plt.subplot(1, 2, 2)
#    plt.imshow(I2)

    #function [Vx,Vy,reliab] = opticalFlow( I1, I2, varargin )
    #% Coarse-to-fine optical flow using Lucas&Kanade or Horn&Schunck.
    #%
    #% Implemented 'type' of optical flow estimation:
    #%  LK: http://en.wikipedia.org/wiki/Lucas-Kanade_method
    #%  HS: http://en.wikipedia.org/wiki/Horn-Schunck_method
    #%  SD: Simple block-based sum of absolute differences flow
    #% LK is a local, fast method (the implementation is fully vectorized).
    #% HS is a global, slower method (an SSE implementation is provided).
    #% SD is a simple but potentially expensive approach.
    #%
    #% Common parameters: 'smooth' determines smoothing prior to computing flow
    #% and can make flow estimation more robust. 'filt' determines amount of
    #% median filtering of the computed flow field which improves results but is
    #% costly. 'minScale' and 'maxScale' control image scales in the pyramid.
    #% Setting 'maxScale'<1 results in faster but lower quality results, e.g.
    #% maxScale=.5 makes flow computation about 4x faster. Method specific
    #% parameters: 'radius' controls window size (and smoothness of flow) for LK
    #% and SD. 'nBlock' determines number of blocks tested in each direction for
    #% SD, computation time is O(nBlock^2). For HS, 'alpha' controls tradeoff
    #% between data and smoothness term (and smoothness of flow) and 'nIter'
    #% determines number of gradient decent steps.
    #%
    #% USAGE
    #%  [Vx,Vy,reliab] = opticalFlow( I1, I2, pFlow )
    #%
    #% INPUTS
    #%  I1, I2   - input images to calculate flow between
    #%  pFlow    - parameters (struct or name/value pairs)
    #%   .type       - ['LK'] may be 'LK', 'HS' or 'SD'
    #%   .smooth     - [1] smoothing radius for triangle filter (may be 0)
    #%   .filt       - [0] median filtering radius for smoothing flow field
    #%   .minScale   - [1/64] minimum pyramid scale (must be a power of 2)
    #%   .maxScale   - [1] maximum pyramid scale (must be a power of 2)
    #%   .radius     - [10] integration radius for weighted window [LK/SD only]
    #%   .nBlock     - [5] number of tested blocks [SD only]
    #%   .alpha      - [1] smoothness constraint [HS only]
    #%   .nIter      - [250] number of iterations [HS only]
    #%
    #% OUTPUTS
    #%  Vx, Vy   - x,y components of flow  [Vx>0->right, Vy>0->down]
    #%  reliab   - reliability of flow in given window
    #%
    #% EXAMPLE - compute LK flow on test images
    #%  load opticalFlowTest;
    #%  [Vx,Vy]=opticalFlow(I1,I2,'smooth',1,'radius',10,'type','LK');
    #%  figure(1); im(I1); figure(2); im(I2);
    #%  figure(3); im([Vx Vy]); colormap jet;
    #%
    #% EXAMPLE - rectify I1 to I2 using computed flow
    #%  load opticalFlowTest;
    #%  [Vx,Vy]=opticalFlow(I1,I2,'smooth',1,'radius',10,'type','LK');
    #%  I1=imtransform2(I1,[],'vs',-Vx,'us',-Vy,'pad','replicate');
    #%  figure(1); im(I1); figure(2); im(I2);
    #%
    #% EXAMPLE - compare LK/HS/SD flows
    #%  load opticalFlowTest;
    #%  prm={'smooth',1,'radius',10,'alpha',20,'nIter',250,'type'};
    #%  tic, [Vx1,Vy1]=opticalFlow(I1,I2,prm{:},'LK'); toc
    #%  tic, [Vx2,Vy2]=opticalFlow(I1,I2,prm{:},'HS'); toc
    #%  tic, [Vx3,Vy3]=opticalFlow(I1,I2,prm{:},'SD','minScale',1); toc
    #%  figure(1); im([Vx1 Vy1; Vx2 Vy2; Vx3 Vy3]); colormap jet;
    #%
    #% See also convTri, imtransform2, medfilt2
    #%
    #% Piotr's Computer Vision Matlab Toolbox      Version 3.50
    #% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
    #% Licensed under the Simplified BSD License [see external/bsd.txt]

    #% run optical flow in coarse to fine fashion
    if(I1.dtype != np.single):
        I1=np.single(I1)
        I2=np.single(I2)
    h , w = I1.shape[0], I1.shape[1]
    nScales=np.max([1, np.int(np.floor( np.log2(np.min([h,w,1/minScale]))))+1])
    
    maxScaleRange = max(1,nScales + np.int(np.round(np.log2(maxScale))))
    
    rangePyr = range(0,maxScaleRange)
    if(reversedPyr): rangePyr = reversed(rangePyr)
    
    for idx, sc in enumerate(rangePyr):
    #  % get current scale and I1s and I2s at given scale
        scale=2**(sc)
        h1=int(round(h/scale))
        w1=int(round(w/scale))
        
        if( scale==1 ):
            I1s=I1
            I2s=I2
        else:
            I1s=cv2.resize(I1,(w1, h1))
            I2s=cv2.resize(I2,(w1, h1))

    #  % initialize Vx,Vy or upsample from previous scale
        if(idx==0):
            Vx=np.zeros((h1,w1))
            Vy=np.zeros((h1,w1))
        else:
            r=np.sqrt(h1*w1/Vx.size)
            Vx=cv2.resize(Vx,(w1, h1))*r
            Vy=cv2.resize(Vy,(w1, h1))*r
    #  % transform I2s according to current estimate of Vx and Vy
        if(idx>1):
            I2s = warp_flow(I2s, np.stack((Vx, Vy), axis=2), factor=-1.0)
#            I2s,__= warp_flow2(I2s, np.stack((Vx, Vy), axis=2), factor=-1.0,method=0)
#            I2s=imtransform2(I2s,[],'pad','replciate','vs',Vx,'us',Vy)
    #  % smooth images
        gaussh = matlab_style_gauss2D([3,3], smooth);
        I1s = convolve(I1s,gaussh, mode='nearest');
        I2s = convolve(I2s,gaussh, mode='nearest');
    #  % run optical flow on current scale
        Vx1,Vy1,reliab=opflowFunction(I1s,I2s,**kwargs)
        
#        plt.figure()
#        plt.subplot(1, 2, 1)
#        plt.imshow(Vx1)
#        plt.subplot(1, 2, 2)
#        plt.imshow(Vy1)

        Vx=Vx+Vx1
        Vy=Vy+Vy1
        
        if(filt): 
            Vx=medfilt2d(Vx,[filt, filt])
            Vy=medfilt2d(Vy,[filt, filt])
    #  % finally median filter the resulting flow field
    r=np.sqrt(h*w/Vx.size);
    if(r != 1):
        Vx=cv2.resize(Vx,(w, h))*r
        Vy=cv2.resize(Vy,(w, h))*r
        reliab=cv2.resize(reliab,(w, h))
#    if(r != 1):
    return Vx,Vy,reliab