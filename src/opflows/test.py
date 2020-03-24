import cv2
from skimage.io import imread
import numpy as np
from decode import decode_optical_flow
from visualization import colorflow_white,arrow_flow
import matplotlib.pyplot as plt
from coarse2fine_flow import coarse2fine_flow
from interpolation.horn_schunck import opticalFlowHS, opticalFlowHSPyr
from interpolation.tvl1 import tvl1_simple, opticalFlowTVL1Pyr
from interpolation.lucas_kanade import opticalFlowLK, opticalFlowLKPyr
import flowmetrics
import block_matching
from copy import deepcopy
from tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS

def main():
    
    
    method = ['block_matching', 'coarse2fine', 'DenseCV', 'HS', 'TVL', 'LK', 'PWCNet']
    
    sel_method = 6
    pyr = False
    
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
    elif sel_method == 6:
        
        gpu_devices = ['/device:CPU:0']  
        controller = '/device:CPU:0'
        
        ckpt_path = '/Users/sergi/mcv-m6-2020-team5/src/opflows/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
        
        nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = True
        nn_opts['ckpt_path'] = ckpt_path
        nn_opts['batch_size'] = 1
        nn_opts['gpu_devices'] = gpu_devices
        nn_opts['controller'] = controller
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
        nn_opts['adapt_info'] = (1, 376, 1241, 2)
        
        nn = ModelPWCNet(mode='test', options=nn_opts)
        nn.print_config()
        
        img_pairs = []
        img_pairs.append((cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB),cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)))
        print(np.asarray(img_pairs).shape)
        
        flow = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
        
        flow = np.asarray(flow)
        flow = np.squeeze(flow,axis = 0)
    else:
        print("El método seleccionado no es válido.")            
    
    if type(flow) is tuple:
        movsx, movsy, reliab = flow
        flow = np.stack((movsx, movsy), axis=2)    
        
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

