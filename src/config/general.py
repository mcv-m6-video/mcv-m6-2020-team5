# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:39:40 2020

@author: hamdd
"""
from .utils import AttrDict
 
STOP_AT = -1
VISUALIZE = True

video = AttrDict()
video.save_video = False
video.fname = 'output_best_adaptative_mean.avi'
video.fps = 30
video.stack_iou = False
video.start_save = 650
video.duration = 3      #duration in seconds

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
detector.dtype = "gauss_black_rem"       # Detector to use
detector.activate_mask = False          # Whether or not to activate the mask to discard possible noise
detector.mask_path = "./img/scene_mask.png"  # path to the mask
detector.backgrounds = AttrDict()
detector.backgrounds.ours = AttrDict()
detector.backgrounds.ours.init_at = 10 # where to init computing IoU and mAP, after training the background
detector.backgrounds.ours.alpha = 5 #Try for different values (2.5 should be good)
detector.backgrounds.ours.rho = 0.01 #If different than 0 then adaptive
detector.backgrounds.ours.color_space ="BGR"
detector.backgrounds.ours.single_channel = "GRAY"

gtruth = AttrDict()
gtruth.src = None # No hace falta cambiar este parametro
gtruth.include_static_iou = False #funcion del sergio para descartar IoU==1
gtruth.include_parked = True    # descartamos aquellos que tienen el atributo parked = 1
gtruth.include_occluded = True # descartamos aquellos que tienen el atributo occluded = "true"