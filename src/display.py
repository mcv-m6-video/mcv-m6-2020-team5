# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:07:14 2020

@author: hamdd
"""
import cv2
import numpy as np


def print_rect(frame, rect, color):
    pt1 = (int(rect[0]), int(rect[1]))
    pt2 = (int(rect[2]), int(rect[3]))
    cv2.rectangle(frame, pt1, pt2, color, thickness=2)

def print_mask_overlap(frame, mask, alpha, color):
    binary = cv2.bitwise_and(mask,mask).astype(bool)
    # binary = np.dstack([binary]*3)
    fcopy = frame.copy()
    frame[binary] = color
    beta = 1-alpha
    frame = cv2.addWeighted(fcopy, beta, frame, alpha, 0.0)
    return frame


def print_single_path(path, frame,  color=(0,255,0)):
     for index, item in enumerate(path[:-1]): 
        cv2.line(frame, item, path[index + 1], color, 4)    
        
def print_func(frame, gt_rects, dt_rects, bgseg, bgseg_o, config, tracking):
    if(gt_rects is not None):
        for gtrect in gt_rects:
            if(config.bboxes.activate):
                if(config.bboxes.gt):
                        print_rect(frame, gtrect, config.bboxes.gt_color)
    if(bgseg_o is not None and config.bgseg_o.activate):
            frame = print_mask_overlap(frame, bgseg_o, 
                               config.bgseg_o.alpha, 
                               config.bgseg_o.color)
    if(bgseg is not None and config.bgseg.activate):
            frame = print_mask_overlap(frame, bgseg, 
                               config.bgseg.alpha, 
                               config.bgseg.color)
    for dt_id, dtrect in dt_rects.items():
        if(config.bboxes.dt):
            print_rect(frame, dtrect, config.bboxes.dt_color)
        if(config.paths.activate):
            print_single_path(tracking.object_paths[dt_id], frame, config.paths.color)


    return frame