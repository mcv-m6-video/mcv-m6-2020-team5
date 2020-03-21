import cv2
import numpy as np


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
