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
    def __init__(self, img, nSplits, w_padding=0, include_padding=False):
        self.img = img
        self.nSplits = int(nSplits)
        self.nRows = int(nSplits)
        self.nCols = int(nSplits)
        self.w_padding = w_padding
        # self.s_w_padding = simulated_w_padding
        if(w_padding>=0.5):
            raise(ValueError("Padding cannot be higher than 0.5: got ",w_padding))
        # Dimensions of the image
        self.size = img.shape[0]
        self.w_padding = int(self.size*w_padding)
        self.start_xy = self.w_padding
        self.psize = int((self.size-self.start_xy*2)/self.nSplits)
        
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
        
        # print(f"executing {self._row} {self._col}, not bigger than {self.nCols}")
        yp = self._col*self.psize
        xp = self._row*self.psize
        
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
        # print(self._row, self._col,yp,self.size,self.start_xy)
        if(m_xhgh <= self.img.shape[1]):
            if(m_yhgh <= self.img.shape[0]):
                roi = self.img[ylow:yhgh, xlow:xhgh]
                # print(roi.shape)
                # cv2.imshow("roi", roi)
                # cv2.waitKey(0)
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
    if(np.count_nonzero(patch1) and np.count_nonzero(patch2)):
        red_corr = conv2(rotate180(patch1).astype(np.float), 
                                   patch2.astype(np.float))
        maxx, maxy = (np.where(red_corr==red_corr.max())[0][0], 
                  np.where(red_corr==red_corr.max())[1][0])
        # diffx = (sx2-sx1) - maxx
        # diffy = (sx2-sx1) - maxy
        return maxx, maxy
    return patch1.shape[0],patch1.shape[1]

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
        # e1 = skimage.measure.shannon_entropy(p1)
        # e2 = skimage.measure.shannon_entropy(p2)
        # if(e1 < 6 or e2 < 6): continue
        # print(f"Entropy: {e1} {e2}")
        movx, movy = obtain_correlation_mov(p1, p2, canny=canny)
        movsx.append(movx)
        movsy.append(movy)
        stack_imgs = np.hstack((p1, p2))
        # cv2.imshow("res",stack_imgs)
        # cv2.waitKey(1)
    return pi_prev.psize-np.mean(movsx), pi_next.psize-np.mean(movsy)

def obtain_dense_mov(img_prev, img_next,
                    area_search = 0.0,
                    black_match_func = obtain_correlation_mov,
                    window_size=0.05, canny=True):
    # if(area_search < window_size):
    #     raise(ValueError("Area of search must be bigger than window size"))
    movsx = np.zeros((img_prev.shape[:2]))
    movsy = np.zeros((img_prev.shape[:2]))
    
    splits = int(1/window_size)
    
    pi_prev = squarePatchIterator(img_prev, splits,area_search, True)
    pi_next = squarePatchIterator(img_next, splits,area_search,False)
    pi_movsx = squarePatchIterator(movsx, splits,area_search,False)
    pi_movsy = squarePatchIterator(movsy, splits,area_search,False)
    
    # movsx = np.zeros((pi_prev.get_n_patches()))
    # movsy = np.zeros((pi_prev.get_n_patches()))
    # movsx = np.zeros
    # start_xy = pi_prev.start_xy
    # movsx = []
    # movsy = []
    # movsx_l = []
    # movsy_l = []
    for p1, p2, mx, my in zip(pi_prev, pi_next, pi_movsx, pi_movsy):  
        movx, movy = obtain_correlation_mov(p1, p2, canny=canny)
        fx = pi_prev.area_sz-movx
        fy = pi_prev.area_sz-movy
        mx[:] = fx
        my[:] = fy
        # print(f"MOV X: {fx} MOV Y: {fy}")
        # movsx_l.append(movx)
        # movsy_l.append(movy)
        # movsx.append(movx)
        # movsy.append(movy)
        # stack_imgs = np.hstack((p1, p2))
        # cv2.imshow("res",stack_imgs)
        # cv2.waitKey()
    # m_movx, m_movy = pi_prev.psize-np.mean(movsx), pi_next.psize-np.mean(movsy)
    
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
        
        # img_next_z = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
        # img_next_p = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
        movx, movy = fix_strategy(img_prev_z, img_next_z, canny=canny)
        accx = accx if max_mov is not None and accx >= max_mov else accx+movx
        accy = accy if max_mov is not None and accy >= max_mov else accy+movy
        # s = f"ACC:{accx},{accy}  MOV:{movx},{movy}"
        # print(s)
        # print(movx, movy)
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
    # y1 = 0
    # x1 = int(center[0] - int(height/2))
    # y2 = int(y1+height)
    # x2 = int(x1+height)

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
        
        
        # img_next_z = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
        # img_next_p = cv2.resize(img_next_p, None, fx=nzoom, fy=nzoom) 
        flow = dense_strategy(img_next_p, img_next_p, 
                              window_size=window_size,
                              area_search = area_search, 
                              canny=canny)
        flow *= flow*inversezoom
        # accx = accx if max_mov is not None and accx >= max_mov else accx+movx
        # accy = accy if max_mov is not None and accy >= max_mov else accy+movy
        # s = f"ACC:{accx},{accy}  MOV:{movx},{movy}"
        # print(s)
        # print(movx, movy)
        
        # M = np.float32([[1,0,accy*inversezoom],[0,1,accx*inversezoom]])
        
        # img_next_w = cv2.warpAffine(img_next, M, (width, height))
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
    pbar = tqdm(desc="Frame", total=frame_count)
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
    # fix_video(fpath)
    view_dense(fpath, nzoom=1, window_size=0.01, area_search=0.03, canny=True)