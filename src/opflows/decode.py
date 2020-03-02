import cv2
import numpy as np



def decode_optical_flow(im):
    im_float = im.astype("double")
    w,h = im_float.shape[:2]
    flow_im = np.zeros((w,h,2))
    valid_flow =np.zeros((w,h))
    flow_im[:,:,0] = (im_float[:,:,2]-pow(2,15))/64.0
    flow_im[:,:,1] = (im_float[:,:,1]-pow(2,15))/64.0

    valid_flow  = im_float[:,:,0] > 0
    
    return flow_im,valid_flow


if __name__ == "__main__":
    im = cv2.imread("/Users/sergi/mcv-m6-2020-team5/datasets/results/LKflow_000045_10.png", cv2.IMREAD_UNCHANGED )

    flow_im,valid_flow = decode_optical_flow(im)