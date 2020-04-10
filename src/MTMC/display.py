#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:12:59 2020

@author: dazmer
"""
import numpy as np
import cv2
import pickle

def display_min(dist_mat):
    nzeros = np.zeros_like(dist_mat)
    nzeros[np.arange(0, dist_mat.shape[0]),dist_mat.argmin(axis=1)]=1
    rgb = cv2.cvtColor(nzeros*255,cv2.COLOR_GRAY2RGB)
    return rgb

def display_heatmap(dist_mat):
    
    # row_sums = dist_mat.sum(axis=1)
    # new_matrix = (dist_mat / row_sums[:, np.newaxis]).numpy()
    
    # row_maxs = dist_mat.max(axis=1)[0]
    # new_matrix = (dist_mat / row_maxs[:, np.newaxis]).numpy() 
    
    row_maxs = np.max(dist_mat.numpy())
    new_matrix = (dist_mat.numpy() / row_maxs)
    
    new_matrix -= np.mean(new_matrix)
    new_matrix[new_matrix<0] = 0

    row_maxs = np.max(new_matrix)
    new_matrix = (new_matrix / row_maxs)
        
    
    new_matrix *= 255
    new_matrix = new_matrix.astype(np.uint8)
    
    new_matrix = 255-new_matrix
    # gray = cv2.cvtColor(new_matrix, cv2.COLOR_RGB2GRAY)
    out = cv2.applyColorMap(new_matrix, cv2.COLORMAP_RAINBOW)
    # cv2.imshow("res", out); cv2.waitKey(1000)
    return out

def print_grid(img, p1, p2, color=(0, 0, 255)):
    # path1 = os.path.normpath(pickle1) 
    # path2 = os.path.normpath(pickle2) 

    
    counter = 0
    for k in p1.keys():
        counter += len(p1[k])
        img[counter-1, :,:] = color

    counter = 0
    for k in p2.keys():
        counter += len(p2[k])
        # try:
        img[:,counter-1,:] = color
        # except:
        #     print("out of bounds!")
        #     print(counter,"-",len(p2[k])-1)

displaying = False
sel_x, sel_y = -1, -1
def show_pair_imgs(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        displaying = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if displaying == True:
            pass
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
