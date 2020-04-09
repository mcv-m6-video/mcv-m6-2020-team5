#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:28:52 2020

@author: dazmer
"""
from torchreid import metrics
import pickle
import numpy as np
import torch

def create_matrix(p1, p2, dist_metric="euclidean"):
    # f1 = []
    # for k in p1.keys():
    #     f1.extend(p1[k])
    # f2 = []
    # for k in p2.keys():
    #     f2.extend(p2[k])
    f1 = [p1[k] for k in p1.keys()]
    f2 = [p2[k] for k in p2.keys()]
    
    f1 = np.squeeze(np.concatenate(f1,axis=0),axis=1)
    f2 = np.squeeze(np.concatenate(f2,axis=0),axis=1)
    
    p1 = torch.tensor(f1)
    p2 = torch.tensor(f2)

    distances = metrics.compute_distance_matrix(p1, p2, dist_metric)
    return distances

def calculate_matrices(cam_pickles):
    relation_cams = {}
    for cam1 in cam_pickles.keys():
        relation_cams[cam1] = {}
        for cam2 in cam_pickles.keys():
            if(cam2 == cam1): continue
            p1 = pickle.load(open(cam_pickles[cam1], "rb"))
            p2 = pickle.load(open(cam_pickles[cam2], "rb"))
        
            relation_cams[cam1][cam2] = create_matrix(p1, p2)
    return relation_cams
        