#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:28:12 2020

@author: dazmer
"""
import torchreid
from torchvision import transforms
import cv2
import torch 

class Model(object):
    def __init__(self,
                 name = "resnet50",
                 weights = "./log/resnet50_ai2019_triplet/1/model.pth.tar-2"):
        self.model = torchreid.models.build_model(
                                name=name,
                                num_classes=500,#Asignar a prueba error, de momento
                                loss='triplet',
                                pretrained=True
                                # use_gpu = True
                                )
        self.model = self.model.cuda()
        torchreid.utils.load_pretrained_weights(self.model, weights)
        self.transform = transforms.Compose([
                            # transforms.Resize(( 200,200 )),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    def forward(self, img):
        out = self.model(img)
        chrcs = out[1].cpu().detach().numpy()
        return chrcs
    def __call__(self, img):
        if(type(img)==str):
            img = cv2.imread(img)
        img = cv2.resize(img, (200, 200))
        img_ex = self.transform(img)
        img_ex = img_ex.cuda()
        batch = torch.unsqueeze(img_ex, 0)
        return self.forward(batch)