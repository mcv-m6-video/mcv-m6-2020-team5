# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:39:40 2020

@author: hamdd
"""
from .utils import AttrDict
 
STOP_AT = -1
VISUALIZE = True
WEIGHTS = "../weights"

video = AttrDict()
video.save_video = False
video.fname = 'output_best_adaptative_mean.avi'
video.fps = 30
video.stack_iou = True
video.start_save = 3000
video.duration = 3      #duration in seconds

tracker = AttrDict()
tracker.ttype = "centroid"
tracker.Multi = AttrDict()
tracker.Multi.maxDisappeared=0
tracker.Multi.pix_tol=100
tracker.Sort = AttrDict()
tracker.Sort.max_age=7
tracker.Sort.min_hits=2


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
pout.paths = AttrDict()
pout.paths.activate = True
pout.paths.color = (0, 255, 0)

plots = AttrDict()
plots.iou = AttrDict()
plots.iou.name = "IoU_frame"        # Name of the plot
plots.iou.max_val = int(30*2.5)     # Max value of the X axis
plots.iou.save = False              # Save every frame of the plot
plots.iou.n_frames = 1              # Save every N frames

detector = AttrDict()
detector.dtype = "detectron"       # Detector to use
detector.activate_mask = False          # Whether or not to activate the mask to discard possible noise
detector.mask_path = "./img/scene_mask.png"  # path to the mask
detector.backgrounds = AttrDict()
detector.backgrounds.ours = AttrDict()
detector.backgrounds.ours.init_at = 0 # where to init computing IoU and mAP, after training the background
detector.backgrounds.ours.alpha = 5 #Try for different values (2.5 should be good)
detector.backgrounds.ours.rho = 0.01 #If different than 0 then adaptive
detector.backgrounds.ours.color_space ="BGR"
detector.backgrounds.ours.single_channel = "GRAY"
detector.detectron = AttrDict()
detector.detectron.train_frames = 2140
detector.detectron.weights_path = WEIGHTS+"/detectron.weights"
detector.detectron.net = 'faster_rcnn' # Possible neta: retinanet, faster_rcnn
detector.detectron.training = True
detector.detectron.train_method = 'random' # Possible methods (so far): random25 (25% random), initial, random50 (50% random)

bbox_filter = AttrDict()
bbox_filter.wf_high=0.4
bbox_filter.hf_high=0.4
bbox_filter.wf_low=0.005
bbox_filter.hf_low=0.005 
bbox_filter.min_size=0.0003
bbox_filter.max_size=0.1
bbox_filter.form_factor_low=0.2
bbox_filter.form_factor_high=10



gtruth = AttrDict()
gtruth.src = None # No hace falta cambiar este parametro
gtruth.include_static_gt = True # if false, we do not use the static gt bounding boxes using IoU measure
gtruth.include_parked = True    # if false, we do not use the static gt bounding boxes using attribute park = 1
gtruth.include_occluded = True # if false, we do not use the gt bounding boxes using attribute occluded = "true"

