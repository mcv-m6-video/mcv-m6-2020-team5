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
import skimage.measure
from visualization import colorflow_white, colorflow_black
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
    def __init__(self, img, nSplits, 
                 w_padding=0, include_padding=False,
                 step_size=None, include_step=False):
        self.img = img
        self.nSplits = int(nSplits)
        self.nRows = int(nSplits)
        self.nCols = int(nSplits)
        self.w_padding = w_padding

        if(w_padding>=0.5):
            raise(ValueError("Padding cannot be higher than 0.5: got ",w_padding))

        self.size = img.shape[0]
        self.w_padding = int(self.size*w_padding)
        self.start_xy = self.w_padding
        
        remaining_space = (self.size-self.start_xy*2)
        
        if(step_size is None):
            if(include_step):
                raise(ValueError(f"include_step is set, but got step_size: {step_size}"))
            include_step = False
            step_size = 1
            fpsize = int(remaining_space/(self.nSplits))
            spsize = int(remaining_space/(self.nSplits/step_size))
            
        elif(step_size == 0):
            fpsize = int(remaining_space/(self.nSplits))
            spsize = 1
        else:
            fpsize = int(remaining_space/(self.nSplits))
            spsize = int(remaining_space/(self.nSplits/step_size))
            # if(fpsize+spsize == 0):
            #     raise(ValueError("Step size too slow!, try using another aproximation"))
        if(include_step):
            nstep_size = 1 if step_size is None else step_size
            if(nstep_size > 1): 
                raise(ValueError("step_size must be lower than 1"))
            self.psize = fpsize
            self.r_size = 0
        else:
            self.psize = spsize
            self.r_size = fpsize - spsize
        if(spsize == 0):
            raise(ValueError("Step size resultant is zero!! try increasing step_size"))
        self.step_size = spsize
        
        self.include_step = include_step

        self._row = 0
        self._col = 0
        
        self._next_x_start = self.start_xy
        self._next_y_start = self.start_xy
        
        self.include_padding = include_padding
        
        self.area_sz = self.psize+self.w_padding*2
    def get_n_patches(self):
        return self.nRows*self.nCols
    def __iter__(self):
        return self
    def __next__(self):
        
        yp = self._col*self.step_size
        xp = self._row*self.step_size
        
        ylow = self.start_xy+yp
        xlow = self.start_xy+xp
        
        yhgh = ylow + self.psize
        xhgh = xlow + self.psize
        
        m_ylow = ylow-self.w_padding
        m_xlow = xlow-self.w_padding
        m_yhgh = yhgh+self.w_padding
        m_xhgh = xhgh+self.w_padding
        
        if(self.include_padding):
            ylow = m_ylow
            yhgh = m_yhgh
            xlow = m_xlow
            xhgh = m_xhgh
        m_ylow += self.r_size
        m_xlow += self.r_size
        m_yhgh += self.r_size
        m_xhgh += self.r_size

        if(m_xhgh <= self.img.shape[1]):
            if(m_yhgh <= self.img.shape[0]):
                roi = self.img[ylow:yhgh, xlow:xhgh]

                self._col+=1
                return roi 
            else:
                self._col = 0
                self._row +=1
        else:
            raise(StopIteration())
        
        return self.__next__()

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
    hl = int(red_corr.shape[0]/2)+1
    maxx, maxy = hl, hl
    if(np.count_nonzero(patch1) and np.count_nonzero(patch2)):
        red_corr = red_corr
        max_val = red_corr.max()
        
        r255 = ((red_corr/max_val)*255).astype(np.uint8)
        cv2.imshow("r255", r255)
        cv2.waitKey(1)
        
        maxx, maxy = np.where(red_corr==max_val)
        maxx, maxy = maxx[0], maxy[0]

    return -(maxx-hl), -(maxy-hl)


def obtain_mean_mov_squared(img_prev, img_next, 
                            block_match_func = obtain_correlation_mov,
                            window_size=0.25, canny=True):
    splits = int(1/window_size)
    pi_prev = squarePatchIterator(img_prev, splits)
    pi_next = squarePatchIterator(img_next, splits)

    movsx = []
    movsy = []
    for p1, p2 in zip(pi_prev, pi_next):  

        movx, movy = obtain_correlation_mov(p1, p2, canny=canny)
        movsx.append(movx)
        movsy.append(movy)
        stack_imgs = np.hstack((p1, p2))

    return np.mean(movsx), np.mean(movsy)

def obtain_dense_mov(img_prev, img_next,
                    area_search = 0.0,
                    black_match_func = obtain_correlation_mov,
                    window_size=0.05,
                    step_size=None,
                    canny=True):
    
    movsx = np.zeros((img_prev.shape[:2]))
    movsy = np.zeros((img_prev.shape[:2]))
    
    splits = int(1/window_size)
    
    pi_prev = squarePatchIterator(img_prev, splits,area_search,True ,step_size, True)
    pi_next = squarePatchIterator(img_next, splits,area_search,False,step_size, True)
    
    pi_movsx = squarePatchIterator(movsx, splits, area_search,False,step_size, False)
    pi_movsy = squarePatchIterator(movsy, splits, area_search,False,step_size, False)

    for p1, p2, mx, my in zip(pi_prev, pi_next, pi_movsx, pi_movsy):  
        movx, movy = obtain_correlation_mov(p1, p2, canny=canny)
        
        fx = movx
        fy = movy
        mx[:] = fx
        my[:] = fy
    flow = np.stack((movsx, movsy), axis=2)
    return flow

    
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
              canny=True, fix_strategy = obtain_mean_mov_squared,
              max_mov=20, get_video=True):
    inversezoom = 1/nzoom 
   
    out_cap = None
    cap = cv2.VideoCapture(videopath)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, img_prev = cap.read()
    
    
    center = (width/2.0, height/2.0)
    y1 = 0
    x1 = int(center[0] - int(height/2))
    y2 = int(y1+height)
    x2 = int(x1+height)

    img_prev_p = img_prev[y1:y2, x1:x2, 0]
    img_prev_z = cv2.resize(img_prev_p, None, fx=nzoom, fy=nzoom) 
    
    ret, img_next = cap.read()
    
    pbar = tqdm(desc="Matrices calc", total=frame_count)
    accx = 0
    accy = 0
    i=0
    while cap.isOpened() and ret:
        i+=1
        pbar.update()
        img_next_p = img_next[y1:y2, x1:x2, 0]
        img_next_z = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 

        movx, movy = fix_strategy(img_prev_z, img_next_z, canny=canny)

        accx = accx if max_mov is not None and accx >= max_mov else accx+movx
        accy = accy if max_mov is not None and accy >= max_mov else accy+movy

        M = np.float32([[1,0,accy*inversezoom],[0,1,accx*inversezoom]])
        img_next_w = cv2.warpAffine(img_next, M, (width, height))
        cv2.imshow("Wrapped", img_next_w)
        cv2.waitKey(1)
        
        f_out = np.hstack((img_next, img_next_w))
        if(out_cap is None  and get_video):
            fshape = f_out.shape
            out_cap = cv2.VideoWriter("out.avi", 
                                    cv2.VideoWriter_fourcc(*"MJPG"), 
                                    fps, 
                                    (fshape[1],fshape[0]))
        out_cap.write(f_out.astype('uint8'))
        img_next = img_next_w
        img_prev_z = img_next_z
        ret, img_next = cap.read()
    cap.release()
    if(out_cap is not None): 
        out_cap.release()
    
def view_dense(videopath, nzoom = 0.3, window_size = 0.25,
              canny=True, dense_strategy = obtain_dense_mov,
              area_search = 0.0,
              max_mov=20, get_video=True):
    inversezoom = 1/nzoom 
   
    out_cap = None
    cap = cv2.VideoCapture(videopath)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, img_prev = cap.read()
    
    
    center = (width/2.0, height/2.0)

    img_prev_p = img_prev[:,:, 0]
    img_prev_z = cv2.resize(img_prev_p, None, fx=nzoom, fy=nzoom) 
    
    ret, img_next = cap.read()
    
    pbar = tqdm(desc="Matrices calc", total=frame_count)
    accx = 0
    accy = 0
    i=0
    # zx1, zx2, zy1, zy2 = map(lambda x: int(x*nzoom), [x1, x2, y1, y2])
    while cap.isOpened() and ret:
        i+=1
        pbar.update()
        
        
        img_next_z = cv2.resize(img_next, None, fx=nzoom, fy=nzoom) 
        z_placeholder = np.zeros_like(img_next_z)
        img_next_p = img_next_z[:,:, 0]

        flow = dense_strategy(img_next_p, img_next_p, 
                              window_size=window_size,
                              area_search = area_search, 
                              canny=canny)
        flow *= flow*inversezoom

        rgb = colorflow_white(flow)
        cv2.waitKey(1)
        z_placeholder[:,:] = rgb
        f_out = np.hstack((img_next_z, z_placeholder))
        cv2.imshow("out", f_out)
        if(out_cap is None  and get_video):
            fshape = f_out.shape
            out_cap = cv2.VideoWriter("out.avi", 
                                    cv2.VideoWriter_fourcc(*"MJPG"), 
                                    fps, 
                                    (fshape[1],fshape[0]))
        out_cap.write(f_out.astype('uint8'))
        # img_next = img_next_w
        img_prev_p = img_next_p
        ret, img_next = cap.read()
    cap.release()
    if(out_cap is not None): 
        out_cap.release()
        

if __name__ == "__main__":
    fpath = "/home/dazmer/Videos/non_stabilized5.mp4"
    fix_video(fpath, max_mov=None)
    # view_dense(fpath, nzoom=1, window_size=0.021, area_search=0, canny=True)