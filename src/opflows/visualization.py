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
    rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    # rgb = rgb.astype(np.uint8)
    return rgb

def arrow_flow(vects,im):
    scale = 4
    vects_resized = cv2.resize(vects,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_LINEAR)
    u = vects_resized[:,:,0]
    v = vects_resized[:,:,1]
    y = np.arange(0, im.shape[0], 1)
    x = np.arange(0, im.shape[1], 1)
    x, y = np.meshgrid(x, y)
    
    idxi = []
    idxj = []
    
    uu = []
    vv = []
    
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if u[i,j] > 0 and v[i,j] > 0: 
                
                idxi.append(i*scale)
                idxj.append(j*scale)
                
                uu.append(2*scale*u[i,j])
                vv.append(2*scale*v[i,j])
                
                
    # s=3
    # u = vects[0:-1:s, 0:-1:s,0]
    # v = vects[0:-1:s, 0:-1:s,1]
    
    norm_im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.figure()
    plt.imshow(norm_im) #Posar-la en el rang que toca
    plt.quiver(idxj, idxi, uu, vv, scale_units='xy', angles='xy', scale = 1., color='w')
    plt.show()
    plt.savefig('plot1.png')
    
    #Don't print arrows with 0's