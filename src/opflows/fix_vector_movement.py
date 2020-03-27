#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:55:29 2020

@author: dazmer
"""
import os
import cv2
import pickle
import numpy as np
import glob
from visualization import colorflow_white,arrow_flow,get_plot_legend
from decode import decode_optical_flow
import tqdm
from collections import OrderedDict
import flowmetrics

import csv
# a = pickle.load(open("output/flow_matrix_0.005_0.01_1.pkl", "rb"))

# b = a[:,:,(1,0)]

# output_folder = "output/"
imgspaths = glob.glob("output/*.pkl")

output_folder = "output_corrected2/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

curr_dir = "../../"
gt_p = curr_dir+"/datasets/of_pred/noc_000045_10.png"

gt = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED)


#flow_im,valid_flow = decode_optical_flow(im)
flow_gt, val_gt_flow = decode_optical_flow(gt)

_m=50

for i_path in tqdm.tqdm(imgspaths):
    r_name = i_path.split("/")[1]
    wz, az, sz = r_name.split("flow_matrix_")[1][:-4].split("_")
    flow = pickle.load(open(i_path, "rb"))
    flow_c = -flow[:,:,(1,0)]
    img = colorflow_white(flow_c)
    fname = f"_{wz}_{az}_{sz}"
    with open(output_folder+r_name, "wb+") as f:
        pickle.dump(flow_c, f)
    cv2.imwrite(output_folder+"rgb_image_"+fname+".png", img)
    msen, pepn = flowmetrics.flowmetrics(flow_c[_m:-_m,_m:-_m,:], flow_gt[_m:-_m,_m:-_m,:], val_gt_flow[_m:-_m,_m:-_m])
    register_vals = OrderedDict()
    register_vals["step_size"]=sz
    register_vals["window_sizes"]=wz
    register_vals["area_search"]=az
    register_vals["msen"]=msen
    register_vals["pepn"]=pepn
                
    with open(output_folder+"registering.csv", "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(register_vals.keys()))
        writer.writerow(register_vals)
    print(f"Result: MSEN={msen} PEPN={pepn}")
                