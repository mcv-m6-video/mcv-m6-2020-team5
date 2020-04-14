#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:51:10 2020

@author: dazmer
"""
import glob
import os
import torchreid
# torchreid.data.ImageData
class AI2019Dataset(torchreid.data.ImageDataset):
    dataset_dir = 'aic20_reID_bboxes'

    def __init__(self, root='', combineall=True, **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.data_dir = os.path.join(self.root, self.dataset_dir)
        print(self.data_dir)

        self.train_dir = os.path.join(self.data_dir,"bounding_box_train")
        self.query_dir = os.path.join(self.data_dir,"bounding_box_test_reduced")
        self.gallery_dir=os.path.join(self.data_dir,"bounding_box_test_reduced")
        
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(AI2019Dataset, self).__init__(train, 
                                            query, 
                                            gallery,
                                            **kwargs)
    def process_dir(self, data_dir, relabel=False):
        img_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        
        pid_container = set()
        for img_path in img_paths:
            # print("IMG PATH:", img_path)
            # print("RSPLIT:", 
            img_name = img_path.rsplit("/", 1)[-1]
            pid, cid, iid = img_name.split("_")
            iid = iid.split(".")[0]
            pid = int(pid)
            # cid = int(cid[1:])
            if(pid == -1):
                 continue # junk images are just ignored
            pid_container.add(pid)
            
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
    
        data = []
        for img_path in img_paths:
            img_name = img_path.rsplit("/", 1)[-1]
            pid, camid, iid = img_name.split("_")
            pid = int(pid)
            camid = int(camid[1:])
            
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
    
    
torchreid.data.register_image_dataset('AI2019_challange', AI2019Dataset)

datamanager = torchreid.data.ImageDataManager(
    root='/media/dazmer/datasets/traffic/AIC20_track3_MTMC/',
    sources='AI2019_challange',
    targets='AI2019_challange',
    height=200,
    width=200,
    batch_size_train=32,
    batch_size_test=32,
    num_instances=4,
    train_sampler='RandomIdentitySampler',
    transforms=['random_flip', 'random_crop']
)

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