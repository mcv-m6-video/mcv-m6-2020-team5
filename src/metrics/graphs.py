
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 12:25:19 2020

@author: hamdd
"""

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import cv2
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

class LinePlot(object):
    def __init__(self,plot_name, max_val = 300, save_plots=False,dir_save="./plots",
                 make_video = True):
        self.line = None
        self.fig = plt.figure(plot_name)
        plt.ion()
        plt.show(block=False)
        self.ax = self.fig.add_subplot(111)
        self.line = None
        self.ax.set_ylim(0,1)
        self.ax.set_xlim(0,max_val)
        plt.xlabel("frame")
        plt.ylabel("IoU")
        # plt.xlim(auto=True)
        self.idx = 0
        self._y_data = [np.nan] * max_val
        self.max_val = max_val
        
        self.save_plots=save_plots
        if(save_plots):
            self.folder_name = f"{dir_save}/{plot_name}/"
            pathlib.Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        self.last_img = None
    def update(self, new_line):
        if self.idx >= self.max_val:
            self._y_data.pop(0)
            self._y_data.append(np.nan)
            self._y_data[-1] = new_line
            # self.ax.set_xlim(self.idx-self.max_val, self.idx)
            # self.ax.set_xticks()
            ticks = np.arange(0, self.max_val+1, 1)
            labels = np.arange(self.idx-self.max_val, self.idx-1, 1)
            labels_mask = np.where(np.logical_not(labels%50))
            ticks = ticks[labels_mask]
            labels = labels[labels_mask]
            plt.xticks(ticks, labels)
        else:
            self._y_data[self.idx] = new_line
        if self.line is None:
            self.line, = self.ax.plot(self._y_data)
        else:
            # self.line.remove()
            self.line.set_ydata(self._y_data)
        # self.line, = plt.plot(new_line)
        self.fig.canvas.draw()
        plt.pause(0.001)
        self.idx+=1
    def build_frame(self, frame):
        w,h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = (h, w, 4)
        f_figure_im = obtain_frame_and_figure(buf, frame)
        self.last_img = f_figure_im
    def save_plot(self, frame=None):
        fname = f'{self.folder_name}{self.idx:4d}.jpg'
        if(frame is not None):
            self.build_frame(frame)
            # w,h = self.fig.canvas.get_width_height()
            # buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8 )
            # buf.shape = (h, w, 4)
            # f_figure_im = obtain_frame_and_figure(buf, frame)
            cv2.imwrite(fname,self.last_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        else:
            self.fig.savefig(fname)
def iouFrame(iou):
    plt.plot(iou)
    plt.ylim(0,1)
    plt.show()

def obtain_frame_and_figure(figure_im, frame):
    fh, fw = frame.shape[:2]
    ih, iw = figure_im.shape[:2]
    iw_final = int(iw*(fh/ih))
    
    f_figure_im = cv2.resize(figure_im, (iw_final, fh))
    f_figure_im = f_figure_im[:,:,(3,2,1)]
    # f_figure_im = f_figure_im[:,:,:3]
    final_figure = np.hstack((f_figure_im, frame))
    return final_figure

def plot_3d_surface(X, Y, Z):

    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),\
                            np.linspace(np.min(Y),np.max(Y),10))
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis') 
    plt.xlabel("Rho")
    plt.ylabel("Alpha")
    plt.title("mAP")
    plt.show()
        
if __name__ == "__main__":
    X = []
    Y = []
    Z = []
    with open('gridsearch.txt', 'r') as f:
        for line in f:
            fields = line.split(",")
            X.append(float(fields[0]))
            Y.append(float(fields[1]))
            Z.append(float(fields[2]))
    
    plot_3d_surface(X,Y,Z)
    