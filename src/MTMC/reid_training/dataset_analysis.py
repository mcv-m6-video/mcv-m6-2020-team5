#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:58:22 2020

@author: dazmer
"""
import matplotlib.pyplot as plt

import os 
import glob

def plot_hist(D):
    ldict = [(k, v) for k, v in D.items()]
    ldict = sorted(ldict, key=lambda x:x[1])
    plt.bar(range(len(D)), list([i[1] for i in ldict]), align='center')
    plt.xticks(range(len(D)), list([i[0] for i in ldict]))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    
    plt.show()
    
def dict_classes(path):
    id_list = {}
    for path in glob.glob(os.path.join(root_path, "*")):
        oid, cid, fname = path.rsplit("/", 1)[1].split("_")
        if oid not in id_list:
            id_list[oid] = {"total":0, "paths":[]}
        id_list[oid]["total"]+=1
        id_list[oid]["paths"].append(path)
    return id_list
        
if __name__ == "__main__":
    root_path = "/media/dazmer/datasets/traffic/AIC20_track3_MTMC/aic20_reID_bboxes/bounding_box_test"

    id_list = dict_classes(root_path)
    
    phist = {k:d["total"] for k, d in id_list.items()}
    plot_hist(phist)
    
    total = sum([d["total"] for d in id_list.values()])
