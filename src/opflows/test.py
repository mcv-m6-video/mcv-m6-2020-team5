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

def main():
    
    
    method = ['block_matching', 'coarse2fine', 'DenseCV', 'HS', 'TVL', 'LK']
    
    sel_method = 4
    pyr = True
    
    gt_paths = ["/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000045_10.png",
                "/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000157_10.png"]
    
    select_image = 0 #0: image 1, 2: image 2
    
    im = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/results/LKflow_000157_10.png", cv2.IMREAD_UNCHANGED)
    
    if select_image == 0:
        im1 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000045_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000045_11.png", cv2.IMREAD_UNCHANGED )
    elif select_image == 1:
        im1 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000157_10.png", cv2.IMREAD_UNCHANGED )
        im2 = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000157_11.png", cv2.IMREAD_UNCHANGED )
    
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
        flow = obtain_dense_mov(im1,im2)
    elif sel_method == 7:
        block_match2 = bm2.EBMA_searcher(15,15)
        im_warped, flow = block_match2.run(im1,im2)
    else:
        print("El método seleccionado no es válido.")            
        
        
    color_plot = colorflow_white(flow)
    cv2.imshow("color_plot",color_plot)
    cv2.waitKey(1)
    
    
    arrow_flow(flow,im1)
    
    ##metrics
    msen, pepn = flowmetrics.flowmetrics(flow, flow_gt, val_gt_flow)
    
    print('MSEN')
    print(msen)
    print('PEPN')
    print(pepn)
    
if __name__ == "__main__":
    main()

