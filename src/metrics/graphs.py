
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

class LinePlot(object):
    def __init__(self,plot_name, max_val = 300, save_plots=False,dir_save="./plots"):
        self.line = None
        self.fig = plt.figure(plot_name)
        plt.ion()
        plt.show(block=False)
        self.ax = self.fig.add_subplot(111)
        self.line = None
        self.ax.set_ylim(0,1)
        self.ax.set_xlim(0,max_val)
        # plt.xlim(auto=True)
        self.idx = 0
        self._y_data = [np.nan] * max_val
        self.max_val = max_val
        
        self.save_plots=save_plots
        if(save_plots):
            self.folder_name = f"{dir_save}/{plot_name}/"
            pathlib.Path(self.folder_name).mkdir(parents=True, exist_ok=True)
    def update(self, new_line):
        if self.idx >= self.max_val:
            self._y_data.pop(0)
            self._y_data.append(np.nan)
            self._y_data[-1] = new_line
            # self.ax.set_xlim(self.idx-self.max_val, self.idx)
            # self.ax.set_xticks()
            ticks = np.arange(0, self.max_val, 1)
            labels = np.arange(self.idx-self.max_val, self.idx, 1)
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
    def save_plot(self, frame=None):
        fname = f'{self.folder_name}{self.idx:4d}.jpg'
        if(frame is not None):
            w,h = self.fig.canvas.get_width_height()
            buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8 )
            buf.shape = (h, w, 4)
            f_figure_im = obtain_frame_and_figure(buf, frame)
            cv2.imwrite(fname,f_figure_im, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
    f_figure_im = f_figure_im[:,:,:3]
    final_figure = np.hstack((f_figure_im, frame))
    cv2.imshow("test", final_figure)
    return final_figure

