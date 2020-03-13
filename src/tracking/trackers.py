#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:00:38 2020

@author: dazmer
"""
from .centroid_tracker import CentroidTracker


def get_centroid(rect):
    x1, y1, x2, y2 = rect
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x, y)


def obtain_tracker(ttype, config):
    if(ttype=="centroid"):
        tracker = CentroidTracker(key=get_centroid, **config.centroid)
    else:
        raise(ValueError(f"Tracker type not recognized: {ttype}"))
    return tracker