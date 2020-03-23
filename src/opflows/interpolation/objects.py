# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:58:50 2018

@author: hamdd
"""

from .utils import Enumeration

class nImagesList(Enumeration):
    def __init__(self):
        self.TWO_IMG = 0

class entryTypeList(Enumeration):
    def __init__(self):
        self.GRAY = 0
        self.ANY = 1