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
from MTSC.test_detectron import test_detectron
from MTSC.train_detectron import train_detectron


    
if __name__ == "__main__":
    parser = general_parser()
    args = parser.parse_args()
    new_gconfig = []
    configs_jj = []
    if args.general_config is not None:
        new_gconfig.extend(args.general_config)

    # train_detectron(new_gconfig)
    test_detectron(new_gconfig)
    
