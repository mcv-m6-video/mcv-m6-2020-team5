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

# def star_mask(patch, color):
#     for x in range(patch.shape[0]):
#         patch[x,x,:] = color
#         patch[x,-x,:] = color
def display_heatmap(dist_mat, display_peaks = True, peak_color=(0,255,165)):

   
    
    # row_sums = dist_mat.sum(axis=1)
    # new_matrix = (dist_mat / row_sums[:, np.newaxis]).numpy()
    
    # row_maxs = dist_mat.max(axis=1)[0]
    # new_matrix = (dist_mat / row_maxs[:, np.newaxis]).numpy() 
    
    row_maxs = np.max(dist_mat.numpy())
    new_matrix = (dist_mat.numpy() / row_maxs)
    
    # means_row = np.array(new_matrix.mean(1))
    mean_total = new_matrix.mean()
    
    new_matrix[new_matrix>mean_total] = mean_total
    new_matrix -= new_matrix.min()
    
    row_maxs = np.max(new_matrix)
    new_matrix = (new_matrix / row_maxs)
        
    
    new_matrix *= 255
    new_matrix = new_matrix.astype(np.uint8)
    
    new_matrix = 255-new_matrix
    # gray = cv2.cvtColor(new_matrix, cv2.COLOR_RGB2GRAY)
    out = cv2.applyColorMap(new_matrix, cv2.COLORMAP_OCEAN)
    if(display_peaks):
        out[np.arange(0, dist_mat.shape[0]),dist_mat.argmin(axis=1),:]=peak_color

    # cv2.imshow("res", out); cv2.waitKey(1000)
    return out

sel_x, sel_y = -1, -1
displaying = False
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

def print_axis(img):
    icpy = img.copy()
    global displaying, sel_x, sel_y
    if(displaying and sel_x > -1 and sel_y > -1):
        if(sel_y < img.shape[0] and sel_x < img.shape[1]):
            icpy[sel_y, :,:] = (255, 0, 0)
            icpy[:, sel_x,:] = (255, 0, 0)
        
    return icpy

def show_pair_imgs(event,x,y,flags,param):
    global displaying, sel_x, sel_y
    if event == cv2.EVENT_LBUTTONDOWN:
        displaying = True
        # ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if displaying == True:
            pass
    elif event == cv2.EVENT_LBUTTONUP:
        displaying = False
    
    if displaying:
        sel_x, sel_y = x, y
        # c1 = param[0]
        # c2 = param[1]
        # # print("C1:", len(c1), "x:", x)
        # # img1 = c1[y]
        # # img2 = c2[x]
        # # print("img1:", img1)
        # cv2.imshow("car1", img1)
        # cv2.imshow("car2", img2)
        # cv2.waitKey(100)
    # else:
        # sel_x, sel_y = -1, -1
def obtain_xy():
    global sel_x, sel_y 
    return sel_x, sel_y




from collections import OrderedDict
def view_cars_matrix(mat_relations, img_pickles):
    exited = False
    i = 0
    while not exited and i < len(mat_relations):
        tracklet = mat_relations[i]
        exited, plus_or_minus = view_cars_tracklet(tracklet, img_pickles)
        i += plus_or_minus
        if(i < 0): i = 0
        if(i >= len(mat_relations)):
            i = len(mat_relations)-1
            print("No more tracklets to display!")
        
def view_cars_tracklet(tracklet, img_pickles):
    print("Tracklet:", tracklet)
    c_imgs = OrderedDict()
    for car_id, (cam_id, pickle_path) in zip(tracklet,  img_pickles.items()):
        p_c = pickle.load(open(pickle_path, "rb"))
        if(car_id != -1):
            imgs = p_c[int(car_id)]
            del p_c
            c_imgs[cam_id] = imgs
    # print("Images extracted:", c_imgs)
    i=0
    exited = False
    another_tracklet = False
    plus_or_minus = +1
    print("press 'q' to exit")
    print("press 'a' to go to prev image")
    print("press 'd' to go to next image")
    print("press 'w' to go to prev tracklet")
    print("press 's' to go to next tracklet")
    while not exited and not another_tracklet:
        # imgs_to_show = OrderedDict()
        for k in c_imgs.keys():
            if(i < len(c_imgs[k])):
                cv2.imshow(str(k), c_imgs[k][i])
            
        k = cv2.waitKey(0)
        if(k != -1):
            print("key:", k)
        if(k == ord("d")):
            i+=1
        if(k == ord("a")):
            i-=1
        if(k == ord("w")):
            another_tracklet = True
            plus_or_minus = -1
        if(k == ord("s")):
            another_tracklet = True
            plus_or_minus = +1
        if(k == ord("q")):
            exited = True
        if(i < 0):
            i = 0
    return exited, plus_or_minus
            
            
            
            
            
            