# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:48:59 2020

@author: hamdd
"""

import argparse

def add_general_config(parser):
    parser.add_argument('--general_config', type=str, metavar="PARAM[.SUBPARAM[...]]",nargs='+',
                    help='Parameters that will change the default general \
                    configuration file', default=None)
                        
def general_parser():
    parser = argparse.ArgumentParser(description='Analyzes the video given the desired parameters')
    add_general_config(parser)
    return parser