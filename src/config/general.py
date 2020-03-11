# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:39:40 2020

@author: hamdd
"""
from .utils import AttrDict
 
STOP_AT = -1
VISUALIZE = True

video = AttrDict()
video.save_video = True
video.fname = 'output_pickup2.avi'
video.stack_iou = False
video.start_save = 1700
video.end_save = 2140


# PRINT CONFIGURATION
pout = AttrDict()           #Print Out, to avoid re-declare "print" function
pout.activate = True            # Whether activate or not prints
pout.bboxes = AttrDict()
pout.bboxes.activate = True     # visualize bboxes?
pout.bboxes.gt = True           # visualize bboxes of gt?
pout.bboxes.dt = True           # visualize bboxes of detections?
pout.bboxes.gt_color = (255, 0, 0) # color of bboxes of gt
pout.bboxes.dt_color = (0, 255, 0) # color of bboxes of detections
pout.bgseg = AttrDict()
pout.bgseg.activate = True      # visualize background segmentation?
pout.bgseg.color = (255, 0, 255)# color of background segmentation
pout.bgseg.alpha = 0.75         # alpha of background segmentation
pout.bgseg_o = AttrDict()
pout.bgseg_o.activate = True        # visualize original background segmentation?
pout.bgseg_o.color = (0, 255, 255)  # color of original background segmentation
pout.bgseg_o.alpha = 0.7            # alpha of original background segmentation


plots = AttrDict()
plots.iou = AttrDict()
plots.iou.name = "IoU_frame"        # Name of the plot
plots.iou.max_val = int(30*2.5)     # Max value of the X axis
plots.iou.save = False              # Save every frame of the plot
plots.iou.n_frames = 1              # Save every N frames

detector = AttrDict()
detector.type = "gauss_black_rem"       # Detector to use
detector.activate_mask = False          # Whether or not to activate the mask to discard possible noise
detector.mask = "./img/scene_mask.png"  # path to the mask
detector.init_at = 535 # where to init computing IoU and mAP, after training the background
detector.alpha = 4 #Try for different values (2.5 should be good)
detector.rho = 0 #If different than 0 then adaptive


gtruth = AttrDict()
gtruth.static = True
