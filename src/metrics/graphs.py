# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 12:25:19 2020

@author: hamdd
"""

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

class LinePlot(object):
    def __init__(self,plot_name, max_val = 300):
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
        
    def update(self, new_line):
        if self.idx >= self.max_val:
            self._y_data.pop(0)
            self._y_data.append(np.nan)
            self._y_data[-1] = new_line
            # self.ax.set_xlim(self.idx-self.max_val, self.idx)
            # self.ax.set_xticks()
            ticks = np.arange(0, self.max_val, 1)
            labels = np.arange(self.idx-self.max_val, self.idx, 1)
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
        
def iouFrame(iou):
    plt.plot(iou)
    plt.ylim(0,1)
    plt.show()
    