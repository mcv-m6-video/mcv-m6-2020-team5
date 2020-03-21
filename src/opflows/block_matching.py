# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:11:10 2018
@author: hamdd
"""

import glob
import cv2
from scipy.signal import convolve2d as conv2

# import pylab as pl
import numpy as np

from os import path, mkdir
import re
from tqdm import tqdm 

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class squarePatchIterator(object):
    def __init__(self, img, nSplits):
        self.img = img
        self.nSplits = int(nSplits)
        self.nRows = int(nSplits)
        self.nCols = int(nSplits)

        # Dimensions of the image
        self.size = img.shape[1]
        self.psize = int(self.size/self.nSplits)
        # self.sizeY = img.shape[0]

        self._row = 0
        self._col = 0
    def get_n_patches(self):
        return self.nRows*self.nCols
    def __iter__(self):
        return self
    def __next__(self):
        print(f"executing {self._row} {self._col}, not bigger than {self.nCols}")
        
        if(self._row < self.nRows):
            if(self._col < self.nCols):
                ylow = self._col*self.psize
                yhgh = ylow + self.psize
                
                xlow = self._row*self.psize
                xhgh = xlow + self.psize
                roi = self.img[ylow:yhgh, xlow:xhgh]
                cv2.imshow("roi", roi)
                # cv2.waitKey(0)
                self._col+=1
                return roi 
            else:
                self._col = 0
                self._row +=1
        else:
            raise(StopIteration())
        return self.__next__()



cv2.waitKey()

def rotate180(img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def obtain_correlation_mov(patch1, patch2, canny=True):
    if(canny):
        patch1 = cv2.Canny(patch1, 10, 70)
        patch2 = cv2.Canny(patch2, 10, 70)
    red_corr = conv2(rotate180(patch1).astype(np.float), 
                               patch2.astype(np.float))
    maxx, maxy = (np.where(red_corr==red_corr.max())[0][0], 
              np.where(red_corr==red_corr.max())[1][0])
    # diffx = (sx2-sx1) - maxx
    # diffy = (sx2-sx1) - maxy
    return maxx, maxy

def obtain_mean_mov_squared(img_prev, img_next, 
                            block_match_func = obtain_correlation_mov,
                            window_size=0.25, canny=True):
    splits = int(1/window_size)
    pi_prev = squarePatchIterator(img_prev, splits)
    pi_next = squarePatchIterator(img_next, splits)
    
    # movsx = np.zeros((pi_prev.get_n_patches()))
    # movsy = np.zeros((pi_prev.get_n_patches()))
    movsx = []
    movsy = []
    for p1, p2 in zip(pi_prev, pi_next):
        movx, movy = obtain_correlation_mov(p1, p2, canny=canny)
        movsx.append(movx)
        movsy.append(movy)
    return np.mean(movsx), np.mean(movsy)
        
def obtain_mov_just_for_center(img_prev, img_next, 
                               block_match_func = obtain_correlation_mov,
                               window_size=0.25, canny=True):
    
    # height, width = img_prev.shape[:2]

    # img_prev_p = cv2.resize(img_prev_p, None, fx=nzoom, fy=nzoom) 
    
    nheight, nwidth = img_prev.shape
    
    new_center = int(nwidth/2), int(nheight/2)
    sx1 = int(new_center[1]-int(nheight*window_size)) #Tocar això fa moure la finestra en vertical
    sy1 = int(new_center[0]-int(nheight*window_size)) #Tocar això fa moure la finestra en horitzontal
    sx2 = int(new_center[1]+int(nheight*window_size)) #Tocar això fa moure la finestra en vertical
    sy2 = int(new_center[0]+int(nheight*window_size)) #Tocar això fa moure la finestra en horitzontal
    
       

    # img_next_p = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
    img_prev_p_c = img_prev[sx1:sx2, sy1:sy2]
    img_next_p_c = img_next[sx1:sx2, sy1:sy2]
    maxx, maxy = obtain_correlation_mov(img_prev_p_c, img_next_p_c, canny=canny)
    movx = ((sx2-sx1) - maxx)
    movy = ((sx2-sx1) - maxy)
    return movx, movy
    
def fix_video(videopath, nzoom = 0.3, window_size = 0.25,
              canny=True, fix_strategy = obtain_mov_just_for_center):
    inversezoom = 1/nzoom 
   
    
    cap = cv2.VideoCapture(videopath)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, img_prev = cap.read()
    ret, img_next = cap.read()
    
    img_prev_z = cv2.resize(img_prev, None, fx=nzoom, fy=nzoom) 
    center = (width/2.0, height/2.0)
    y1 = 0
    x1 = int(center[0] - int(height/2))
    y2 = int(y1+height)
    x2 = int(x1+height)

    img_prev_p = img_prev_z[y1:y2, x1:x2, 0]
    
    img_prev_z = cv2.resize(img_prev, None, fx=nzoom, fy=nzoom) 
    img_next_p = img_prev_z[y1:y2, x1:x2, 0]
    pbar = tqdm(desc="Matrices calc", total=frame_count)
    while cap.isOpened() and ret:
        pbar.update()
        img_next_z = cv2.resize(img_next, None, fx=nzoom, fy=nzoom) 
        # img_next_p = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
        movx, movy = fix_strategy(img_prev_p, img_next_p, canny=canny)
        M = np.float32([[1,0,movy*inversezoom],[0,1,movx*inversezoom]])
        img_next = cv2.warpAffine(img_next, M, (width, height))
        cv2.imshow("Wrapped", img_next)
        cv2.waitKey(100)

        img_prev = img_next
        ret, img_next = cap.read()
    cap.release()

def fix_video2(videopath, nzoom = 0.5, window_size = 0.25, margin = 10): 
    inversezoom = 1/nzoom 
    
    cap = cv2.VideoCapture(videopath)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (width/2.0, height/2.0)
    
    y1 = 0
    x1 = int(center[0] - int(height/2))
    y2 = int(y1+height)
    x2 = int(x1+height)
        

    ret, img_prev = cap.read()
    img_prev = img_prev[y1:y2, x1:x2, 0]
    img_prev = cv2.resize(img_prev, None, fx=nzoom, fy=nzoom) 
    img_prev = cv2.Canny(img_prev, 10, 70)
    
    nheight, nwidth = img_prev.shape
    new_center = int(nwidth/2.0), int(nheight/1.5)
    
    sx1 = int(new_center[1]-int(nheight*window_size)) #Tocar això fa moure la finestra en vertical
    sy1 = int(new_center[0]-int(nheight*window_size)) #Tocar això fa moure la finestra en horitzontal
    sx2 = int(new_center[1]+int(nheight*window_size)) #Tocar això fa moure la finestra en vertical
    sy2 = int(new_center[0]+int(nheight*window_size)) #Tocar això fa moure la finestra en horitzontal
    
    matrices = []
    pbar = tqdm(desc="Matrices calc", total=frame_count)
    ret, img_next = cap.read()
    while cap.isOpened() and ret:
        
        
        
        img_next = img_next[y1:y2, x1:x2, 0]
        img_next = cv2.resize(img_next, None, fx=nzoom, fy=nzoom) 
        img_next = cv2.Canny(img_next, 10, 70)
        
        #TODO Magic edge trick
        pbar.update()
        red_corr = conv2(rotate180(img_next[sx1:sx2, sy1:sy2]), 
                         np.array(img_prev[sx1:sx2,sy1:sy2],dtype=np.float))
        maxx, maxy = (np.where(red_corr==red_corr.max())[0][0], 
                      np.where(red_corr==red_corr.max())[1][0])
        diffx = (sx2-sx1) - maxx
        diffy = (sx2-sx1) - maxy
        M = np.float32([[1,0,-diffy],[0,1,-diffx]])
        matrices.append(M)
        img_dst = cv2.warpAffine(img_next,M,(nwidth,nheight))
    
        cv2.imshow("Where it went", img_dst[sx1:sx2, sy1:sy2])
        cv2.waitKey(1)
        # pl.imshow(img_dst[sx1:sx2, sy1:sy2])
        # pl.title("WHERE IT WENT")
        # pl.pause(.1)
        # pl.draw()
    
        img_prev = img_dst
        ret, img_next = cap.read()
    cap.release()
    cap = cv2.VideoCapture(videopath)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0);

    i=0
    ret=True
    while cap.isOpened() and ret:
        ret, img_next = cap.read()
        cv2.imshow("non-wrapped", img_next)
        M = matrices[i]
        M[0][2] = int(M[0][2]*inversezoom)
        M[1][2] = int(M[1][2]*inversezoom)
        img_next = cv2.warpAffine(img_next, M, (width, height))
        # img_cutted = img_next[margin:(height-margin), margin:(width-margin)]  
        cv2.imshow("Wrapped", img_next)
        cv2.waitKey(100)
        i+=1        

if __name__ == "__main__":
    fpath = "/home/dazmer/Videos/non_stabilized3.mp4"
    fix_video(fpath)