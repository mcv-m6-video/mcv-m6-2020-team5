import cv2
import numpy as np
from decode import decode_optical_flow
from visualization import colorflow_white,arrow_flow
import matplotlib.pyplot as plt
from coarse2fine_flow import coarse2fine_flow
from interpolation.horn_schunck import opticalFlowHS, opticalFlowHSPyr
from interpolation.tvl1 import tvl1_simple, opticalFlowTVL1Pyr
from flowmetrics import flowmetrics
from sklearn.model_selection import ParameterGrid


def process(im1,im2,gt,search_area,block_size)
    
    #flow_im,valid_flow = decode_optical_flow(im)
    flow_gt, val_gt_flow = decode_optical_flow(gt)
    
    #Como llamar al Block Matching
    flow = blockmatching(im1,gt,search_area,block_size)      
        
    ##metrics
    msen, pepn = flowmetrics(flow, flow_gt, val_gt_flow)
    return msen, pepn

if __name__ == "__main__":
    
    select_image = 0 #0: image 1, 2: image 2
    
    gt_paths = ["/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000045_10.png",
                "/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000157_10.png"]    
    
    if select_image == 0:
        im1 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000045_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000045_11.png", cv2.IMREAD_UNCHANGED )
    elif select_image == 1:
        im1 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000157_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000157_11.png", cv2.IMREAD_UNCHANGED )
    
    gt = cv2.imread(gt_paths[select_image], cv2.IMREAD_UNCHANGED)
    
    parameters = {'search_area': [],
                  'block_size':[]}
    
    grid = ParameterGrid(parameters)
    
    msen_list = []
    pepn_list = []
    
    for params in grid:
        print("Params: ", params)
        msen, pepn = process(im1, im2, gt_flow, params["search_area"], params["block_size"] )
        msen_list.append(msen)
        pepn_list.append(pepn)
    
    
    