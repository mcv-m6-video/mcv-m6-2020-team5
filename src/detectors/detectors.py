#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:23:45 2020

@author: dazmer
"""
import cv2
from .backgrounds import BGSTModule
from .groundtruths import gt_predict, gt_yolo_predict, gt_ssd_predict, \
                         gt_rcnn_predict


detectors_dict = {"gt_noise":gt_predict,
             "gt_yolo": gt_yolo_predict,
             "gt_ssd":  gt_ssd_predict,
             "gt_rcnn": gt_rcnn_predict}

_DET_BACKGROUNDS = ["color_gauss_black_rem","gauss_black_rem", "MOG", "MOG2", 
                    "CNT", "GMG", "LSBP", "GSOC", "Subsense", "Lobster"]

_NN_DETECTORS = ["detectron"]

_COLOR_SPACE = ['BGR','RGB','BGRA','RGBA','XYZ','YCBCR','HSV','LAB','LUV',
                'HLS','YUV']
_SINGLE_CHANNEL = ['GRAY','HUE','L','Y','SATURATION'] #Añadir más

def obtain_bgseg_detector(dtype, activate_mask=True, mask_path=True, init_at=10, 
                          alpha=None, rho=None, color_space=None, single_channel=None):
    mask_path = mask_path if activate_mask else None
    cspace = None
    if (dtype in ["color_gauss_black_rem", "gauss_black_rem"]):
        if(alpha is None or rho is None):
            raise(ValueError(f"Alpha and rho not set, got {alpha} and {rho} instead"))
        if(dtype == "color_gauss_black_rem"):
            if(color_space not in _COLOR_SPACE):
                raise(ValueError(f"Color space not recognized: {color_space}, available: {_COLOR_SPACE}"))
            else:
                cspace = color_space
        elif(dtype == "gauss_black_rem"):
            if(single_channel not in _SINGLE_CHANNEL):
                raise(ValueError(f"Single channel of color space not recognized: {single_channel}, available: {_SINGLE_CHANNEL}"))
            else:
                cspace = single_channel
    bgsg_module = BGSTModule(bs_type = dtype, 
                             rho = rho, 
                             alpha = alpha, 
                             init_at = init_at, 
                             color_space = cspace,
                             scene_mask_path=mask_path)
    f = bgsg_module.get_contours
    return f, bgsg_module
            
def obtain_detector(dtype=None, activate_mask=None, 
                    mask_path=None, backgrounds = {}, detectron = {},
                    gt_frames=None):
    bgsg_module = None
    all_detector_names = _DET_BACKGROUNDS + list(detectors_dict.keys()) + _NN_DETECTORS
    if(dtype not in all_detector_names):
        raise(ValueError(f"Detector name '{dtype}' not recognized. Available: {all_detector_names}"))
    if(dtype in _DET_BACKGROUNDS):
        func_detector, bgsg_module = obtain_bgseg_detector(dtype, 
                                              activate_mask=activate_mask, mask_path=mask_path,
                                              **backgrounds.ours)
    if(dtype == "detectron"):
        import detectors.detectron_detect as dt
        dclass = dt.detectron_detector(**detectron, gt_frames=gt_frames)
        func_detector = dclass.predict
    if(dtype in detectors_dict.keys()):
        func_detector = detectors_dict[dtype]
    return func_detector, bgsg_module
   
            
    

    