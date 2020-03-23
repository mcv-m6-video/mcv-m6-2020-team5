# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:45:57 2018

@author: hamdd
"""

from .linear import linear as linear
from .farneback import farneback
from .lucas_kanade import opticalFlowLKPyr
from .horn_schunck import opticalFlowHSPyr
from .tvl1 import tvl1_simple, opticalFlowTVL1Pyr
from .sepconvNN import sepconvNN
from .sepconvNN import sepconvNN_REP
from .sepconvNN import sepconvNN_BOTH
from .sepconvNN import sepconvNN_CONS
from .sepconvNN import sepconvNN_JUMP

from .objects import nImagesList, entryTypeList

nImages = nImagesList()
entryType = entryTypeList()

FLOW = True
NOT_FLOW = False

linear_names = ("linear", "lineal", "l")
farneback_names = ("farneback", "faneback", "f", "franeback")
lucas_kanade_names = ("lucas_kanade", "lucas-kanade", "lk")
horn_schunck_names = ("horn_schunck", "horn_shunck", "hs", "h", "horn-shunck",\
                      "horn-schunk")
tvl1_names = ("tvl1","tv","t","tl","tlv1")
sepconvNN_names = ("sepconvNN","sepconv","sep","conv","nn","sepconvnn","s")
sepconvNN_REP_names = ("s_rep","sr","sRep","srep")
sepconvNN_BOTH_names = ("s_both","sb","sBoth","sboth")
sepconvNN_CONS_names = ("s_cons","sc","sCons","scons")
sepconvNN_JUMP_names = ("s_jump","sj","sJump","sjump")

ifuncDict = {}

__nI = nImages
__eT = entryType

ifuncDict[linear]       = (linear_names,           __nI.TWO_IMG, __eT.ANY,  NOT_FLOW)
ifuncDict[farneback]    = (farneback_names,        __nI.TWO_IMG, __eT.GRAY, FLOW)
ifuncDict[opticalFlowLKPyr] = (lucas_kanade_names, __nI.TWO_IMG, __eT.GRAY, FLOW)
ifuncDict[opticalFlowHSPyr] = (horn_schunck_names, __nI.TWO_IMG, __eT.GRAY, FLOW)
ifuncDict[opticalFlowTVL1Pyr]  = (tvl1_names,      __nI.TWO_IMG, __eT.GRAY, FLOW)
ifuncDict[sepconvNN]      = (sepconvNN_names,      __nI.TWO_IMG, __eT.ANY, NOT_FLOW)
ifuncDict[sepconvNN_REP]  = (sepconvNN_REP_names,  __nI.TWO_IMG, __eT.ANY, NOT_FLOW)
ifuncDict[sepconvNN_BOTH] = (sepconvNN_BOTH_names, __nI.TWO_IMG, __eT.ANY, NOT_FLOW)
ifuncDict[sepconvNN_CONS] = (sepconvNN_CONS_names, __nI.TWO_IMG, __eT.ANY, NOT_FLOW)
ifuncDict[sepconvNN_JUMP] = (sepconvNN_JUMP_names, __nI.TWO_IMG, __eT.ANY, NOT_FLOW)

__all__ = ["linear", "farneback", "opticalFlowLKPyr","opticalFlowHSPyr", \
           "tvl1_simple","sepconvNN"]