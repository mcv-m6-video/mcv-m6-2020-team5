#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:41:18 2020

@author: dazmer
"""
import glob
import torchreid
import os

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
    
def get_datamanager(root_path):
    torchreid.data.register_image_dataset('AI2019_challange', AI2019Dataset)

    datamanager = torchreid.data.ImageDataManager(
        root=root_path,
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
    return datamanager