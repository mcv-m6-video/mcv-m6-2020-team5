import cv2
import numpy as np
from decode import decode_optical_flow
from visualization import colorflow_white,arrow_flow
import matplotlib.pyplot as plt
from coarse2fine_flow import coarse2fine_flow
from interpolation.horn_schunck import opticalFlowHS, opticalFlowHSPyr
from interpolation.tvl1 import tvl1_simple, opticalFlowTVL1Pyr
from interpolation.lucas_kanade import opticalFlowLK, opticalFlowLKPyr
import flowmetrics
import block_matching as bm1
import block_matching2 as bm2 

import os

# Reporting
from collections import OrderedDict
import pickle 
import csv
import time

curr_dir = current_fpath = os.path.realpath(__file__)
curr_dir = curr_dir.rsplit("/", 3)[0]



def main(window_size=0.2,area_search=0.05,step_size  =1):
    
    
    method = ['block_matching', 'coarse2fine', 'DenseCV', 'HS', 'TVL', 'LK']
    
    sel_method = 0
    pyr = True
    
    gt_paths = [curr_dir+"/datasets/of_pred/noc_000045_10.png",
                curr_dir+"/datasets/of_pred/noc_000157_10.png"]
    
    select_image = 0 #0: image 1, 2: image 2
    
    im = cv2.imread(curr_dir+"/datasets/results/LKflow_000157_10.png", cv2.IMREAD_UNCHANGED)
    
    if select_image == 0:
        im1 = cv2.imread(curr_dir+"/datasets/of_pred/000045_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread(curr_dir+"/datasets/of_pred/000045_11.png", cv2.IMREAD_UNCHANGED )
    elif select_image == 1:
        im1 = cv2.imread(curr_dir+"/datasets/of_pred/000157_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread(curr_dir+"/datasets/of_pred/000157_11.png", cv2.IMREAD_UNCHANGED )
    
    gt = cv2.imread(gt_paths[select_image], cv2.IMREAD_UNCHANGED)
    
    
    #flow_im,valid_flow = decode_optical_flow(im)
    flow_gt, val_gt_flow = decode_optical_flow(gt)
    
    
    #Como llamar al Block Matching
    if sel_method == 1:
        flow, im_warped = coarse2fine_flow(im1,im2)
    elif sel_method == 2:
        flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif sel_method == 3:
        if pyr == True:
            flow = opticalFlowHSPyr(im1,im2)
        else:
            flow = opticalFlowHS(im1,im2)
    elif sel_method == 4:
        if pyr == True:
            flow = tvl1_simple(im1,im2)
        else:
            flow = opticalFlowTVL1Pyr(im1,im2)
    elif sel_method == 5:
        if pyr == True:
            flow = opticalFlowLK(im1,im2)
        else:
            flow = opticalFlowLKPyr(im1,im2)
    elif sel_method == 0:
        flow = bm1.obtain_dense_mov(im1,im2,
                                    window_size=window_size,
                                    area_search=area_search,
                                    step_size=step_size)
    elif sel_method == 6:
        block_match2 = bm2.EBMA_searcher(15,15)
    
        im_warped, flow = block_match2.run(im1,im2)
        flow = block_match2.get_original_size(flow,im1.shape[:2])
    else:
        print("El método seleccionado no es válido.")            
        
        
    color_plot = colorflow_white(flow)
    # cv2.imshow("color_plot",color_plot)
    # cv2.waitKey(1)
    
    
    # arrow_flow(flow,im1)
    
    ##metrics
    msen, pepn = flowmetrics.flowmetrics(flow, flow_gt, val_gt_flow)
    
    # print('MSEN')
    # print(msen)
    # print('PEPN')
    # print(pepn)
    return msen, pepn, flow, color_plot
if __name__ == "__main__":
    # step_sizes = [1, 0.5, 0.25, 0.125]
    # window_sizes = [0.2, 0.15, 0.1, 0.05, 0.03, 0.025, 0.02, 0.0125, 0.01]
    # area_search = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.025, 0.02, 0.0125]
    
    step_sizes = [1, 0.5]
    window_sizes = [0.05, 0.025]
    area_search = [0.05, 0.025]
    
    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for sz in step_sizes:
        for wz in window_sizes:
            for az in area_search:
                save_imgs = True
                print("==========================")
                print(f"Executing for w_sz:{wz} a_sx:{az} s_sz:{sz}")
                now = time.time()
                try:
                    msen, pepn, flow, rgb = main(window_size=wz,area_search=az,step_size=sz)
                except:
                    msen = "null"
                    pepn = "null"
                    save_imgs = False
                later = time.time()
                difference = int(later - now)
                if(save_imgs):
                    fname = f"_{wz}_{az}_{sz}"
                    with open(output_folder+"flow_matrix"+fname+".pkl", "wb+") as f:
                        pickle.dump(flow, f)
                    cv2.imwrite(output_folder+"rgb_image"+fname+".png", rgb)
                
                register_vals = OrderedDict()
                register_vals["step_size"]=sz
                register_vals["window_sizes"]=wz
                register_vals["area_search"]=az
                register_vals["msen"]=msen
                register_vals["pepn"]=pepn
                register_vals["time"]=difference
                with open(output_folder+"registering.csv", "a+", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(register_vals.keys()))
                        writer.writerow(register_vals)
                print(f"Result: MSEN={msen} PEPN={pepn}, T_DIFF={difference}")

