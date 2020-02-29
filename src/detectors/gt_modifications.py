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
SOURCE = "../datasets/ai_challenge_s03_c010-full_annotation.xml"

MAX_CHANGE = 1000
STDEV_box_change = 0.01 #Increase noise
PROB_DROPOUT = 0.2
PROB_GENERATE = 0.2
STDEV_box_gen = 0.2

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def obtain_gt(src=None):
    global SOURCE
    src = src if src is not None else SOURCE
    
    with open(src, "rb") as fd:
        gdict = xmltodict.parse(fd)
    frame_list = {}
    for track in gdict["annotations"]["track"]:
        # t_id = track["@id"]
        for box in track["box"]:
            f_id = box["@frame"]
            
            if(f_id not in frame_list):
                frame_list[f_id] = []
            # f_dict[t_id] = {}
            # tmp_fd = f_dict[t_id]
            area = (float(box["@xtl"]), 
                    float(box["@ytl"]),
                    float(box["@xbr"]), 
                    float(box["@ybr"]))
            frame_list[f_id].append(area)
    
    return frame_list
        
def predict(frame):
    global ANN_PER_FRAME
    global _FRAME_ID
    global STDEV
    
    if ANN_PER_FRAME is None:
        ANN_PER_FRAME = obtain_gt(SOURCE)
        
    rects = ANN_PER_FRAME[str(_FRAME_ID)]
    keeped_rects = []
    #delete some rectangles
    for r in rects:
        if(np.random.random() > PROB_DROPOUT):
            keeped_rects.append(r)
    rects = keeped_rects
    
    #Add noise to existant rectangles
    box_changes = get_truncated_normal(0, STDEV_box_change, -1, 1)
    new_rects = []
    for rect in rects:
        mod = box_changes.rvs(4)
        res_area = []
        for a, b in zip(mod, rect):
            res_area.append((a*MAX_CHANGE)+b)
        new_rects.append(res_area)
    rects = new_rects
    #generate random rectangles

    maxh, maxw = frame.shape[:2]
    wh_changes = get_truncated_normal(0, STDEV_box_gen, 0, 1)
    while np.random.random() < PROB_GENERATE:
        x1 = np.random.random()*maxw
        y1 = np.random.random()*maxh
        # x2 = np.random.random()
        # y2 = np.random.random()
        w, h = wh_changes.rvs(2)
        w *= MAX_CHANGE
        h *= MAX_CHANGE
        x2 = x1 + w
        y2 = y1 + h
        
        x2 = x2 if x2 < maxw else maxw
        y2 = y2 if y2 < maxh else maxh
        
        new_rect = (x1, y1, x2, y2)
        rects.append(new_rect)
    _FRAME_ID+=1
    return rects

if __name__ == "__main__":
    source = "../../datasets/ai_challenge_s03_c010-full_annotation.xml"
    dict_frame = generate_dict(source)
    