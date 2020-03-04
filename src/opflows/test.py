<<<<<<< HEAD

import cv2
import numpy as np
from decode import decode_optical_flow
from visualization import color_flow,arrow_flow


def main():
    img_paths = ["../datasets/results/LKflow_000045_10.png",
                 "../datasets/results/LKflow_000157_10.png"]
    select_image = 1
    
    
    
    im = cv2.imread(img_paths[select_image], cv2.IMREAD_UNCHANGED )

    flow_im,valid_flow = decode_optical_flow(im)
    
    color_plot = color_flow(flow_im)
    cv2.imshow("color_plot",color_plot)
    cv2.waitKey(1)
    
    arrow_flow(flow_im,im)


if __name__ == "__main__":
=======

import cv2
import numpy as np
from decode import decode_optical_flow
from visualization import color_flow,arrow_flow


def flowmetrics(pred_flow, gt_flow, thres = 3):
    
    error = pred_flow - gt_flow
    
    npixels = pred_flow.shape[0]*pred_flow.shape[1]
    numberpx = np.sum(np.sum(i > thres for i in error))    
    
    pepn = numberpx/npixels
    
    power = error**2
    summed = np.sum(power)    
    msen = summed/npixels  
    
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
    msen, pepn = flowmetrics(flow_im, flow_gt)
    
    print('MSEN')
    print(msen)
    print('PEPN')
    print(pepn)
    


if __name__ == "__main__":
>>>>>>> 1c222d2fe3e919636c3bd17e3c887a9534de1ca4
    main()