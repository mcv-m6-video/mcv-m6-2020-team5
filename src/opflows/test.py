import cv2
import numpy as np
from decode import decode_optical_flow
from visualization import color_flow,arrow_flow
import matplotlib.pyplot as plt


def flowmetrics(pred_flow, gt_flow, valid_flow, thres = 3):
    
    #Convert flows to vectors and: (pred - gt)^2
    error = (pred_flow[...,:2] - gt_flow[...,:2])**2
    
    error_n = error[valid_flow]
    
    #sqrt((pred-gt)^2)
    mse = np.sqrt(error_n[:,0]+error_n[:,1])
    
    #1/N*sum(sqrt((pred-gt)^2))
    msen = mse.mean()
    
    pepn = 100*(mse > thres).sum()/mse.size
    
    return msen,pepn

def main():
    img_paths = ["/Users/sergi/mcv-m6-2020-team5/datasets/results/LKflow_000045_10.png",
                 "/Users/sergi/mcv-m6-2020-team5/datasets/results/LKflow_000157_10.png"]
    
    gt_paths = ["/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000045_10.png",
                "/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/noc_000157_10.png"]
    
    real_paths = ["/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000045_10.png",
                "/Users/sergi/mcv-m6-2020-team5/datasets/of_pred/000157_10.png"]
    
    select_image = 1 #0: image 1, 1: image 2
    
    
    
    im = cv2.imread(img_paths[select_image], cv2.IMREAD_UNCHANGED )
    gt = cv2.imread(gt_paths[select_image], cv2.IMREAD_UNCHANGED)

    flow_im,valid_flow = decode_optical_flow(im)
    flow_gt, val_gt_flow = decode_optical_flow(gt)
    
    color_plot = color_flow(flow_im)
    cv2.imshow("color_plot",color_plot)
    cv2.waitKey(1)
    
    real_im = cv2.imread(real_paths[select_image], cv2.IMREAD_UNCHANGED)
    arrow_flow(flow_im,real_im)
    
    ##metrics
    msen, pepn = flowmetrics(flow_im, flow_gt, val_gt_flow)
    
    print('MSEN')
    print(msen)
    print('PEPN')
    print(pepn)
    


