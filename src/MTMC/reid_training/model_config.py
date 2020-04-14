#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:51:10 2020

@author: dazmer
"""

from AI2019_dataset import AI2019Dataset, get_datamanager
import os
import torchreid
# torchreid.data.ImageData

DATASET_PATH = '/media/dazmer/datasets/traffic/AIC20_track3_MTMC/'

datamanager = get_datamanager(DATASET_PATH)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    # loss="softmax",
    loss='triplet',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='amsgrad',
    lr=0.001
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

# engine = torchreid.engine.ImageSoftmaxEngine(
#     datamanager,
#     model,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     label_smooth=True
# )
engine = torchreid.engine.ImageTripletEngine(
    datamanager, 
    model, 
    optimizer, 
    margin=0.3,
    weight_t=0.7, 
    weight_x=1, 
    scheduler=scheduler
)
        
def train(model):
    f_path = 'log/resnet50_ai2019/{}'
    i = 0
    done = False
    while not done:
        n_fpath = f_path.format(i)
        if(os.path.exists(n_fpath)):
            i+=1
        else:
            done=True
            os.makedirs(n_fpath)
    
    
    engine.run(
        save_dir=n_fpath,
        max_epoch=10,
        # eval_freq=10,
        print_freq=1,
        test_only=False
    )
    
if __name__ == "__main__":
    r = input("Do you wanna train? y/n")
    if(r=="y"):
        train(model)
    else:
        print("Not training")