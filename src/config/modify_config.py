# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:00:35 2020

@author: hamdd
"""

import importlib

def obtain_str_subsequents(paramstr, list_of_subsequents):
    msg = "{}".format(paramstr)
    for p in list_of_subsequents:
        msg+="[{}]".format(p)
    return msg

def modify_param(gf, param):
    gf_vars = [item for item in dir(gf) if not item.startswith("__")]
    gf_vars = [item for item in gf_vars if not item=="AttrDict"]
    value = True
    #Split between parameter and value
    if("=" in param):
        param, value = param.split("=",1)
        if(value in ["True","False"]):
            value = value =="True"
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except:
                value = str(value)
    # Obtain subparameters (debug.jaume.peris -> debug, jaume.peris)
    subsequent = None
    if( "." in param):
        param, subsequent = param.split(".",1)

    # Obtain that variable
    if(param not in gf_vars):
        msg = "Param '{}' not found in general config".format(param)
        raise(ValueError(msg))
    else:
        var = getattr(gf, param)

    # Set value
    l_of_subsequents = []
    if(subsequent is not None):
        while "." in subsequent:
            sub1, subsequent = subsequent.split(".", 1)
            var = var[sub1]
            l_of_subsequents.append(sub1)
        l_of_subsequents.append(subsequent)
    if(subsequent is not None):
        # print("modifying {}[{}] with {} of type {}".format(param, subsequent, value, type(value)))
        msg = ""
        works = False
        if(not isinstance(var, AttrDict)):
            premsg="You went too far with {}".format(subsequent)
            raise(ValueError(msg))
        elif(subsequent not in var.keys()):
            premsg="Parameter has no attribute {}".format(subsequent)
            errvar = var
        elif(not isinstance(var[subsequent], AttrDict)):
            var[subsequent] = value
            works = True
        else:
            strm = obtain_str_subsequents(param, l_of_subsequents)
            premsg="Selected variable is an entire dict!"
            errvar = var[subsequent]
        if not works:
            print("HELLO")
            strm = obtain_str_subsequents(param, l_of_subsequents)
            msg+= premsg
            msg+="\n  {}".format(strm)
            msg+=" of type {}".format(type(errvar))
            msg+="\n      You can select one of following keys:"
            keys_p = list(errvar.keys())
            for k in keys_p:
                msg+="\n      - {:20s} (of type {})".format(k, type(errvar[k]))
            raise(ValueError(msg))
    else:
        var = value

def modify_config(gf, gconfig):
    if(gconfig is None):
        pass
    else:
        for param in gconfig:
            modify_param(gf, param)

def obtain_general_config(config_path = "config.general", gconfig=None):
    gf = importlib.import_module("config.general")
    modify_config(gf, gconfig)
    return gf