# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:21:28 2020

@author: hamdd
"""

from collections import OrderedDict

from .sort import sort
# tracker = sort.Sort()
import numpy as np

class SortTracker(sort.Sort):
    def __init__(self, max_age=5, min_hits=3, key = lambda x:x):
        super(SortTracker, self).__init__(max_age, min_hits)
        self.objects = OrderedDict()
        self.object_paths = OrderedDict()
        self.key = key
    def update(self, rects):
        self.objects = OrderedDict()
        arr = np.array(rects)
        res = super(SortTracker, self).update(arr)
        # res = self.trackers
        for r in res:
            rid = int(r[-1])
            self.objects[rid] = r[:-1]
            if(rid not in self.object_paths):
                self.object_paths[rid] = []
            self.object_paths[rid].append(self.key(r[:-1]))
        return self.objects