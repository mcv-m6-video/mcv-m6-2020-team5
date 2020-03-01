
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
    main()