# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:39:40 2020

@author: hamdd
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

VISUALIZE = True

video = AttrDict()
video.save_video = True
video.fname = 'output_pickup.avi'
video.stack_iou = False

pout = AttrDict() #Print Out, to avoid re-declare "print" function
pout.activate = True
pout.bboxes = AttrDict()
pout.bboxes.activate = True
pout.bboxes.gt = True
pout.bboxes.dt = True
pout.bboxes.gt_color = (255, 0, 0)
pout.bboxes.dt_color = (0, 255, 0)
pout.bgseg = AttrDict()
pout.bgseg.activate = True
pout.bgseg.color = (255, 0, 255)
pout.bgseg.alpha = 0.75
pout.bgseg_o = AttrDict()
pout.bgseg_o.activate = True
pout.bgseg_o.color = (0, 255, 255)
pout.bgseg_o.alpha = 0.7


plots = AttrDict()
plots.iou = AttrDict()
plots.iou.save = False
plots.iou.name = "IoU_frame"
plots.iou.n_frames = 1
plots.iou.max_val = int(30*2.5)

detector = AttrDict()
detector.type = "MOG"

gtruth = AttrDict()
gtruth.static = True