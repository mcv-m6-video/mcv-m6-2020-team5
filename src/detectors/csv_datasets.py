# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:28 2020

@author: hamdd
"""
import numpy as np
import xmltodict
from scipy.stats import truncnorm

ANN_PER_FRAME = None
_FRAME_ID = 0
SOURCE_RCNN = "../datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
SOURCE_SSD = "../datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"
SOURCE_YOLO = "../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"


def obtain_data(src=None):
    global SOURCE
    src = src if src is not None else SOURCE
    
    frame_dict = {}
    with open(src, "r") as opened_file:
        for line in opened_file:
            line_data = line.split(",")
            
            frame_idx = int(line_data[0])-1 #the first frame starts at 0, not 1
            pt_x = float(line_data[2])
            pt_y = float(line_data[3])
            width = float(line_data[4])
            height = float(line_data[5])
            confidence = float(line_data[6]) 
            
            frame_idx = str(frame_idx)
            if(frame_idx in frame_dict):
                frame_dict[frame_idx].append((pt_x,pt_y,pt_x+width,pt_y+height,confidence))
            else:
                frame_dict[frame_idx] = [(pt_x,pt_y,pt_x+width,pt_y+height,confidence)]
    return frame_dict
        
def predict(frame, src=None):
    global ANN_PER_FRAME
    global _FRAME_ID
    global STDEV
    
    if ANN_PER_FRAME is None:
        ANN_PER_FRAME = obtain_data(src=src)
        
    rects = ANN_PER_FRAME[str(_FRAME_ID)]

    _FRAME_ID+=1
    return rects

def predict_yolo(frame):
    return predict(frame, src=SOURCE_YOLO)
def predict_ssd(frame):
    return predict(frame, src=SOURCE_SSD)
def predict_rcnn(frame):
    return predict(frame, src=SOURCE_RCNN)

if __name__ == "__main__":
    source = "../datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
    dict_frame = obtain_data(source)
    