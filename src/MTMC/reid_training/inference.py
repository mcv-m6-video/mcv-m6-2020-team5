#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:27:52 2020

@author: dazmer
"""

import torch
from torchvision import transforms
import torchreid
from model_config import model, engine
import cv2
import os

transform = transforms.Compose([
    # transforms.Resize(( 200,200 )),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
def obtain_img(path):
    if(type(path)==str):
        img = cv2.imread(path)
    else:
        img = path
    img = cv2.resize(img, (200, 200))
    img_ex = transform(img)
    img_ex = img_ex.cuda()
    batch = torch.unsqueeze(img_ex, 0)
    return batch

class Model(object):
    def __init__(self, type="triplet", 
                 weights = "./log/resnet50_ai2019_triplet/1/model.pth.tar-2"):
        torchreid.utils.load_pretrained_weights(model, weights)
    def forward(self, img):
        out = model(obtain_img(img))
        chrcs = out[1].cpu().detach().numpy()
        return chrcs


if __name__=="__main__":
    root_path = "/media/dazmer/datasets/traffic/AIC20_track3_MTMC/aic20_reID_bboxes/bounding_box_test_reduced"
    p_img1 = os.path.join(root_path, "96_c009_0000005174.jpg")
    p_img2 = os.path.join(root_path,"96_c009_0000005410.jpg")
    p_img3 = os.path.join(root_path,"97_c006_0000000101.jpg")
    p_img4 = os.path.join(root_path,"97_c007_0000013342.jpg")
    
    mdl = Model()
    
    c1 = mdl.forward(p_img1)
    c2 = mdl.forward(p_img2)
    c3 = mdl.forward(p_img3)
    c4 = mdl.forward(p_img4)
    
# engine.run(test_only=True)