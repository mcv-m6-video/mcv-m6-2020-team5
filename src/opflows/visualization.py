import cv2
import numpy as np
import matplotlib.pyplot as plt


# def filter_null_flow(vec):
#     for i in range()

def color_flow(vects):
    w,h = vects.shape[:2]
    hsv = np.zeros((w,h,3))
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(vects[...,0], vects[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # rgb = rgb.astype(np.uint8)
    return rgb

def arrow_flow(vects,im):
    vects_resized = cv2.resize(vects,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    u = vects_resized[:,:,0]
    v = vects_resized[:,:,1]
    y = np.arange(0, im.shape[0], 1)
    x = np.arange(0, im.shape[1], 1)
    x, y = np.meshgrid(x, y)
    # s=3
    # u = vects[0:-1:s, 0:-1:s,0]
    # v = vects[0:-1:s, 0:-1:s,1]

    plt.figure()
    plt.quiver( u, v, scale_units='xy', angles='xy', scale = 1., color='g')
    plt.show()