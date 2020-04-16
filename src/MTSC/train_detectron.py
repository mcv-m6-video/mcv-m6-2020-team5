# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
# import detectors as dts

# from detectors.groundtruths import obtain_gt, obtain_gt_without_static
# from detectors.backgrounds import BGSTModule
from detectors.detectors import obtain_detector
from detectors.groundtruths import obtain_gt


from metrics.mAP import getMetricsClass, IoU
from metrics.graphs import LinePlot, iouFrame
from metrics.mot import mot_metrics
import numpy as np

from display import print_func
from line_arguments import general_parser

from config.utils import obtain_general_config
from detectors.backgrounds.single_gaussian import obtain_global_var_mean
from utils.bbfilters import bbfilters
from metrics.map_all_frames import calculate_ap
from tracking.trackers import obtain_tracker
from tqdm import tqdm 
from collections import OrderedDict
import opflows.visualization as opt_view
from detectors.groundtruths.gt_from_txt import read_gt

import pickle

import glob
import os
import MTSC.detectron_detect_multicameras as dt


def train_detectron(new_config):      
    gconf = obtain_general_config(gconfig=new_config)
    
    # Set get_images_from_video to True to save the frames of the video into a folder for training
    get_images_from_video = False
    
    dclass = dt.detectron_detector_multicameras(create_dataset=get_images_from_video)
    func_detector = dclass.predict