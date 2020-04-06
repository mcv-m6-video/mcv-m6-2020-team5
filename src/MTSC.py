# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
from line_arguments import general_parser


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
    cameras = [10, 11, 12,13,14,15]
    registered = False
    for camera in cameras:
    # train_detectron(new_gconfig)
        test_detectron(new_gconfig,camera, registered)
        registered = True
    
