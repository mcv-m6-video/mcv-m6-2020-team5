# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:39:40 2020

@author: hamdd
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
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
pout.bgseg_o.alpha = 0.4

plots = AttrDict()
plots.iou = AttrDict()
plots.iou.save = True
plots.iou.name = "IoU_frame"
plots.iou.n_frames = 1
plots.iou.max_val = 300

detector = AttrDict()
detector.type = "MOG"

gtruth = AttrDict()
gtruth.static = True