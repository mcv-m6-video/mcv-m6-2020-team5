# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:54:48 2018

@author: hamdd
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 00:47:24 2018

@author: hamdd
"""
import cv2
import numpy as np

import matplotlib.pyplot as plt
from scipy import ndimage
#from matplotlib.transforms import Affine2D
#import mpl_toolkits.axisartist.floating_axes as floating_axes


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def get_gray(img, itype):
    """
    Returns the an image of grays for a given image
    """
    if(itype in ["none", "nc","npy"]):
        sh = img.shape
        if(len(sh) == 3):
            maxh, maxw, d = sh
            if(d == 3):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:,:,0]
        elif(len(sh) == 2):
            gray = img
        else:
            ValueError("Image shape not recognized! (height, width[, depth]), given:", sh)
    elif(itype == "svd"):
        sh = img.shape
        if(len(sh)==2):
            gray = img
        else:
            gray = img[:,:,0]
    elif(itype == "pca"):
        sh = img.shape
        if(len(sh)==2):
            gray = img
        else:
            gray = img[:,:,0]
    elif(itype == "hsv"):
#        gray = cv2.cvtColor(img, cv2.COLOR_HSV2GRAY)
        gray = img[:,:,2]
    elif(itype == "hls"):
#        gray = cv2.cvtColor(img, cv2.COLOR_HLS2GRAY)
        gray = img[:,:,1]
    elif(itype == "ycc"):
        gray = img[:,:,0]
    elif(itype == "xyz"):
        gray = img[:,:,0]
    elif(itype == "lab"):
        gray = img[:,:,0]
    elif(itype == "yuv"):
        gray = img[:,:,0]
    elif(itype == "rgb"):
        gray = img[:,:,1]
    else:
        raise(ValueError("Image Type not recognized:",itype))
    return gray

def of2rgb(u, v):
    hsv = np.zeros(u.shape)
    hsv = np.repeat(hsv[:, :, np.newaxis], 3, axis=2)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(u, v)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    
    return rgb

def obtainRGBScale(step =-0.01):
    first = -1.0
    last = 1.1
    if(step < 0.0):
        first = -first
        last  = -last
    if(step == 0.0):
        raise(ValueError("Step can't be zero!"))
    steps = np.arange(first,last,step)
    xx, yy = np.meshgrid(steps, -steps)
    rgb = of2rgb(xx,yy)
    return rgb

def pltRGBScale(step=-0.01):
    rgb = obtainRGBScale(step=step)
    
#    rgb[:,:,1] = rgb[:,:,1]-1/step
    plt.figure()
    rgb = ndimage.rotate(rgb, 180+90+45)
    plt.imshow(rgb)
#    tr = Affine2D().scale(1, 1).rotate_deg(45)
#
#    start = 0
#    last = (-1/step)*3
#    
##    piv = np.sqrt(np.power(last,2)/2)
##    start -= piv/2
##    last -= piv/2
#    grid_helper = floating_axes.GridHelperCurveLinear(
#        tr, extremes=(start, last, start, last))
#
#    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
#    fig.add_subplot(ax1)
    
#    plt.xticks(rotation=45)

def warp_flow(img, flow, factor=-2.0, borderMode=cv2.BORDER_REPLICATE):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow = -flow
    flow[:,:,0] /=float(factor)
    flow[:,:,1] /=float(factor)
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    flow = np.array(flow, dtype = np.float32)
    if(len(img.shape)>2 and img.shape[2] > 4):
        d = img.shape[2]
        res = np.zeros_like(img)
        for i in range(0,d,4):
            d_low = i
            d_high= min(i+4,d)
#            print(d_low, d_high)
            res[:,:,d_low:d_high] = cv2.remap(img[:,:,d_low:d_high], flow, None, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE )
    else:
        res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderMode = borderMode )
#        res = cv2.remap(img, flow, None, cv2.INTER_CUBIC,  borderMode=cv2.BORDER_TRANSPARENT)
#        res = np.zeros(img.shape)
#        for i in range(h):
#            for j in range(w):
#                newi = i + int(flow[i,j,0])
#                newj = j + int(flow[i,j,1])
#                if newi < h and newi >= 0 and newj < w and newj >= 0:
#                    res[newi,newj,:] = img[i,j,:]
        
    return res

def warp_flow2(img, flow, factor=2.0, method=0, borderMode=cv2.BORDER_REPLICATE):
    h, w = flow.shape[:2]
#    flow = -flow
    flow = flow.copy()
    fac = np.zeros(img.shape)
    if(method==0):
        flow = -flow
        flow[:,:,0] /=float(factor)
        flow[:,:,1] /=float(factor)
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        flow = np.array(flow, dtype = np.float32)
        if(len(img.shape)>2 and img.shape[2] > 4):
            d = img.shape[2]
            res = np.zeros_like(img)
            for i in range(0,d,4):
                d_low = i
                d_high= min(i+4,d)
    #            print(d_low, d_high)
                res[:,:,d_low:d_high] = cv2.remap(img[:,:,d_low:d_high], flow, None, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE )
        else:
#            tile = np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
#            tile = np.reshape(tile, (4,4,1))
#            test = np.tile(tile, (int(img.shape[0]/4), int(img.shape[1]/4), 3))
#            res = cv2.remap(np.uint8(test*255), flow, None, cv2.INTER_CUBIC, borderMode = borderMode )
            res = cv2.remap(img, flow, None, cv2.INTER_CUBIC,  borderMode=borderMode)
    elif(method==1):
#        flow = -flow
        flow[:,:,0] /=float(factor)
        flow[:,:,1] /=float(factor)
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        flow = np.array(flow, dtype = np.float32)
        res = np.zeros(img.shape)
        for i in range(h):
            for j in range(w):
                newi = flow[i,j,1]
                newj = flow[i,j,0]
                gradient = np.array([[0.0,0.0],[0.0,0.0]])
                if newi < h and newi >= 0 and newj < w and newj >= 0:
                    #TODO: Fix borders, when newi >= -1 or <= h, etc. 
                    #gradient inflicts in borders too
                    idxx = int(newi)
                    idxy = int(newj)
                    xdim = newi - idxx
                    ydim = newj - idxy
#                    print("IJ:",i,j,"->",idxx, idxy)
                    gradient[:,0] += 1-xdim
                    gradient[:,1] += xdim
                    gradient[0,:] *= 1-ydim
                    gradient[1,:] *= ydim
                    res[idxx,idxy,:] += gradient[0,0]*img[i, j, :]
                    fac[idxx,idxy,:] += gradient[0,0]
                    if(idxx+1 < h and idxx+1 >= 0): 
                        res[idxx+1,idxy,:] += gradient[1,0]*img[i, j, :]
                        fac[idxx+1,idxy,:] += gradient[1,0]
                    if(idxy+1 < w and idxy+1 >= 0): 
                        res[idxx,idxy+1,:] += gradient[0,1]*img[i, j, :]
                        fac[idxx,idxy+1,:] += gradient[0,1]
                    if(idxx+1 < h and idxy+1 < w and idxx+1 >= 0 and idxy+1 >= 0):
                        res[idxx+1,idxy+1,:] += gradient[1,1]*img[i, j, :]
                        fac[idxx+1,idxy+1,:] += gradient[1,1]
#                    print("PIXEL:",i,j," HAS FLOW:", flow[i,j,0], flow[i,j,1], \
#                          " RESULTING IN:", newi, newj)
#        res[fac>=1.0] /= fac[fac>=1.0]
#        res[res==np.nan] = img[res==np.nan]
        print("finished")
        plt.figure()
#        facnormalized = fac/np.max(fac)
        plt.imshow(fac)
    elif(method==2):
#        flow = -flow
        res = np.zeros(img.shape)
        for i in range(h):
            for j in range(w):
                newi = int(flow[i,j,1])
                newj = int(flow[i,j,0])
                if newi < h and newi >= 0 and newj < w and newj >= 0:
                    res[newi, newj,:] += img[i, j, :]
                    fac[newi, newj,:] += [1.0,1.0,1.0]
        res[fac>=1.0] /= fac[fac>=1.0]
    elif(method==3):
        res = np.zeros(img.shape)
        for i in range(h):
            for j in range(w):
                newi = int(flow[i,j,1])
                newj = int(flow[i,j,0])
                if newi < h and newi >= 0 and newj < w and newj >= 0:
                    res[i, j] += img[newi, newj]
#                    fac[newi, newj,:] += [1.0,1.0,1.0]
#        res[fac>=1.0] /= fac[fac>=1.0]
        
    return np.uint16(res), fac
from scipy.signal import medfilt2d
def votation_flow(flow, factor=2.0, method=0):
    h, w = flow.shape[:2]
    flow = flow.copy()
    fac = np.zeros((h,w))
    if(method == 0):
        flow[:,:,0] /=float(factor)
        flow[:,:,1] /=float(factor)
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
#        flow[:,:,0] += 1.0
#        flow[:,:,1] += 1.0
        flow = np.array(flow, dtype = np.float32)
        for i in range(h):
            for j in range(w):
                newi = flow[i,j,1]
                newj = flow[i,j,0]
                vote = np.array([[0.0,0.0],[0.0,0.0]])
                if newi+1 < h and newi > 0 and newj+1 < w and newj > 0:
                    #TODO: Fix borders, when newi >= -1 or <= h, etc. 
                    #gradient inflicts in borders too
                    idx_x = int(newi)
                    idx_y = int(newj)
                    xdim = newi - idx_x
                    ydim = newj - idx_y
#                    print("IJ:",i,j,"->",idxx, idxy)
                    vote[0,:] += 1-xdim
                    vote[1,:] += xdim
                    vote[:,0] *= 1-ydim
                    vote[:,1] *= ydim
                    fac[idx_x:(idx_x+2),idx_y:(idx_y+2)] += vote
    elif(method == 1):
        flow[:,:,0] /=float(factor)
        flow[:,:,1] /=float(factor)
        npos = flow.copy()
        npos[:,:,0] += np.arange(w)
        npos[:,:,1] += np.arange(h)[:,np.newaxis]
#        flow[:,:,0] += np.arange(w)
#        flow[:,:,1] += np.arange(h)[:,np.newaxis]
#        npos[:,:,0][flow[:,:,0] > 0] += 1.0
#        npos[:,:,0][flow[:,:,0] < 0] -= 1.0
#        npos[:,:,1][flow[:,:,1] > 0] += 1.0
#        npos[:,:,1][flow[:,:,1] < 0] -= 1.0
#        flow = np.array(flow, dtype = np.float32)
        for i in range(h):
            for j in range(w):
                newi = npos[i,j,1]
                newj = npos[i,j,0]
                vote1 = np.array([[0.0,0.0],[0.0,0.0]])
                vote2 = np.array([[0.0,0.0],[0.0,0.0]])
                votef = np.array([[0.0,0.0],[0.0,0.0]])
                if newi+1 < h and newi > 1 and newj+1 < w and newj > 1:
                    #TODO: Fix borders, when newi >= -1 or <= h, etc. 
                    #gradient inflicts in borders too
                    idx_x = int(newi)
                    idx_y = int(newj)
                    xdim = newi - idx_x
                    ydim = newj - idx_y
#                    print("IJ:",i,j,"->",idxx, idxy)
#                    print("POS:", i,j," -->",idx_x, idx_y,"|X", xdim,"Y",ydim)
                    
                    vote1[0,:] += 1-xdim
                    vote1[1,:] += xdim
#                    if(flow[i,j,1]<0.0):
#                        vote1 = np.flip(vote1,0)
#                        idx_x-=2
                        
                    vote2[:,0] += 1-ydim
                    vote2[:,1] += ydim
#                    if(flow[i,j,0]<0.0):
#                        vote2 = np.flip(vote2,1)
#                        idx_y-=2
                    
                    votef = vote1*vote2
                    if(fac[idx_x:(idx_x+2),idx_y:(idx_y+2)].shape != votef.shape):
                        print("wops")
                    fac[idx_x:(idx_x+2),idx_y:(idx_y+2)] += votef
    else:
        raise(ValueError("Method not recognized!"))
    return fac

def test(w=0.0,v=0.0):
    test_flow = np.zeros((10,10,2))
    plt.figure()
    test_flow[:,:,0] = w
    test_flow[:,:,1] = v
    res = votation_flow(test_flow)
    plt.imshow(res)

class Enumeration(object):
    def __init__(self):
        pass
    def __iter__(self):
        li = [getattr(self,attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for x in li:
            yield x
            

#plt.close("all")
#test(0,0)
#test(0.5,0)
#test(1,0)
#test(1.5,0)
#test(2,0)
#test(2.5,0)
#
#test(0,0)
#test(-0.5,0)
#test(-1,0)
#test(-1.5,0)
#test(-2,0)
#test(-2.5,0)
#
#test(0,0)
#test(-0.5,-0.5)
#test(-1,-1)
#test(-1.5,-1.5)
#test(-2,-2)
#test(-2.5,-2.5)
