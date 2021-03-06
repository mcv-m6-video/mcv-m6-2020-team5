# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
import detectors as dts

from detectors.gt_modifications import obtain_gt, obtain_gt_without_static
from detectors.backgrounds import BGSTModule
from metrics.mAP import getMetricsClass
from metrics.graphs import LinePlot, iouFrame
import numpy as np

from display import print_func

from config import general as gconf

from sklearn import svm, datasets
from sklearn.model_selection import ParameterGrid
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"

detectors = {"gt_noise":dts.gt_predict,
             "yolo": dts.yolo_predict,
             "ssd":  dts.ssd_predict,
             "rcnn": dts.rcnn_predict}

parameters = {'rho': [0.8], 
              'alpha':[1]}
# parameters = {'rho': [0.05], 
#               'alpha':[4]}

def gridSearch(rho,alpha):
    INIT_AT = 535 # where to init computing IoU and mAP, after training the background
    STOP_AT = -1
    RHO = rho #If different than 0 then adaptive
    ALPHA = alpha #Try for different values (2.5 should be good)
    DELETE_STATIC_OBJECTS = gconf.gtruth.static # True: deletes static objects from ground truth

    DETECTOR = "gauss_black_rem"
    det_backgrounds = ["color_gauss_black_rem","gauss_black_rem", "MOG", "MOG2", "CNT", "GMG", "LSBP", "GSOC", "Subsense", "Lobster"]
    
    COLOR_SPACE = ['BGR','RGB','BGRA','RGBA','XYZ','YCBCR','HSV','LAB','LUV','HLS','YUV']
    cspace = COLOR_SPACE[0]
    
    SINGLE_CHANNEL = ['GRAY','HUE','L','Y','SATURATION'] #Añadir más
    schannel = SINGLE_CHANNEL[0]
    
    bgsg_module = None

    if(DETECTOR in det_backgrounds):
        if (DETECTOR == "color_gauss_black_rem"):
            bgsg_module = BGSTModule(bs_type = DETECTOR, rho = RHO, alpha = ALPHA, init_at = INIT_AT, color_space = cspace)
        elif (DETECTOR == "gauss_black_rem"):
            bgsg_module = BGSTModule(bs_type = DETECTOR, rho = RHO, alpha = ALPHA, init_at = INIT_AT, color_space = schannel)
        else:
            bgsg_module = BGSTModule(bs_type = DETECTOR, rho = RHO, alpha = ALPHA, init_at = INIT_AT)
        f = bgsg_module.get_contours
        for d in det_backgrounds:
            detectors[d] = f
            
    
    cap = cv2.VideoCapture(SOURCE)
    # cap.set(cv2.CAP_PROP_POS_FRAMES,1450)
    # ret, frame = cap.read()
    if DELETE_STATIC_OBJECTS:
        gt_frames = obtain_gt_without_static()
    else:
        gt_frames = obtain_gt()
        
    i = 0
    avg_precision = []
    iou_history = []
    iou_plot = LinePlot(gconf.plots.iou.name,
                        max_val=gconf.plots.iou.max_val,
                        save_plots=gconf.plots.iou.save)
    # mAP_plot = LinePlot("mAP_frame",max_val=350)
    detect_func = detectors[DETECTOR]
    
    while(cap.isOpened() and (STOP_AT == -1 or i <= STOP_AT)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            #Classify the result
            dt_rects = detect_func(frame)
            
            #Compute the metrics
            avg_precision_frame, iou_frame = getMetricsClass(dt_rects, gt_frames[str(i)], nclasses=1)
            if i > INIT_AT:
                avg_precision.append(avg_precision_frame)
                iou_history.append(iou_frame)
                # iou_plot.update(iou_frame)
            
            #Print Results
 
            gt_rects = gt_frames[str(i)]
            bgseg = None if bgsg_module is None else bgsg_module.get_bgseg()
            orig_bgseg = None if bgsg_module is None else bgsg_module.get_orig_bgseg()

            frame = print_func(frame.copy(), gt_rects, dt_rects, bgseg, orig_bgseg, gconf.pout)
            # cv2.imshow('Frame',frame)
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            i+=1
        # Break the loop
        else:
            break
    print("mIoU for all the video: ", np.mean(iou_history))
    print("mAP for all the video: ", np.mean(avg_precision))
    cap.release()
    return np.mean(avg_precision)
    
    
if __name__ == "__main__":
    
    grid = ParameterGrid(parameters)
    X = []
    Y = []
    Z = []
    n = 0
    for params in grid:
        print("Params: ", params)
        avg_precision = gridSearch(params['rho'], params['alpha'])
        Z.append(avg_precision)
        X.append(params['rho'])
        Y.append(params['alpha'])
        n += 1
    
    # Write the values into a file (rho-alpha-map)
    
    with open('gridsearchtask2.txt', 'w') as f:
        for (rho,alpha,avg_prec) in zip(X,Y,Z):
            f.write("{0},{1},{2}\n".format(rho,alpha,avg_prec))
