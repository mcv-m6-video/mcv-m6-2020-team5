#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:00:38 2020

@author: dazmer
"""
from .multi_tracker import MultiTracker
# from .box_tracker import BoxTracker


def get_centroid(rect):
    x1, y1, x2, y2 = rect
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x, y)

def get_rect(rect):
    return rect



def obtain_tracker(ttype, config):
    if(config.ttype in ["centroid", "overlap"]):
        tracker = MultiTracker(key=get_centroid, **config)
    # elif(config.ttype=="overlap"):
    #     tracker = MultiTracker(key=get_centroid, **config.centroid)
    else:
        raise(ValueError(f"Tracker type not recognized: {ttype}"))
    return tracker