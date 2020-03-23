# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:45:56 2018

@author: hamdd
"""

import paramiko
import json
from os import path, mkdir, remove
#from pprint import pprint
import cv2
import numpy as np
    

def update_num_proc(TextFolder):
    '''
    Function that updates de index of your metadata issue.
    '''
    #Creating root folder
    if not path.exists(TextFolder):
        mkdir(TextFolder)
    
    #Creating metadata
    if not path.isfile(TextFolder+"/metadata.txt"):
        metadata = open(TextFolder+"/metadata.txt", 'w+')
        metadata.write(str(1))
        metadata.close()
    
    metadata = open(TextFolder+"/metadata.txt", 'r+')
    
    num_proc = int([x for x in metadata][0])
    
    metadata.seek(0, 0);  
    metadata.write(str(num_proc+1))
    metadata.close()
    
    return num_proc

def sepconvNN(img1, img2, **kwargs):
    """
    Simple sepconvNN with ANY image
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    #Save images before sending them
    cv2.imwrite("image1.png", img1)
    cv2.imwrite("image2.png", img2)
    
    #Trasnfer Images
    sftp = ssh.open_sftp()
    sftp.chdir(remote)
    path_img1 = "./images/"+name_img1+str(n_exec)+".png"
    path_img2 = "./images/"+name_img2+str(n_exec)+".png"
    print("Uploading image 1...")
    sftp.put("image1.png", path_img1)
    print("Uploading image 2...")
    sftp.put("image2.png", path_img2)
    
    output_img = "/result"+str(n_exec)+".png"
    path_imgi = output_img
    
    #Execute NN
    
    command_execute = "cd "+remote+"\n"
    command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
    command_execute += "export PATH\n"
    command_execute += "source activate sepconv\n"
    command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
    command_execute += "export CUDA_HOME\n"
    command_execute += "CUDA_VISIBLE_DEVICES=1\n"
    command_execute += "export CUDA_VISIBLE_DEVICES\n"
#    command_execute += "conda info --envs\n"
    command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                        " --second "+path_img2+" --out ./"+path_imgi+"\n"
    command_execute += "rm "+path_img1
    command_execute += "rm "+path_img2
    
    print("Running interpolation")
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
    ssh_stdout.channel.recv_exit_status()
    
    #Get interpolated image
    print("Downloading interpolated image...")
    sftp.get("./"+path_imgi, "./"+path_imgi)
    imgi = cv2.imread("./"+path_imgi)
    remove("./"+path_imgi)
    
    #Finish 
    command_execute = "cd "+remote+"\n"
    command_execute+= "rm "+path_imgi
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
    ssh_stdout.channel.recv_exit_status()
    ssh.close()
    
    return imgi

def sepconvNN_REP(img1, img2, **kwargs):
    """
    Runs Sepconv of MS imagery
    MODE: Replicate every band. 
    BORDERS: First and Last eliminated
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    _,_,d = img1.shape
    new_img = list()
    for band in range(1,d-1):
        #Save images before sending them
        cv2.imwrite("image1.png", img1[:,:,(band,band,band)])
        cv2.imwrite("image2.png", img2[:,:,(band,band,band)])
        
        #Trasnfer Images
        sftp = ssh.open_sftp()
        sftp.chdir(remote)
        path_img1 = "./images/"+name_img1+str(n_exec)+".png"
        path_img2 = "./images/"+name_img2+str(n_exec)+".png"
        print("Uploading image 1...")
        sftp.put("image1.png", path_img1)
        print("Uploading image 2...")
        sftp.put("image2.png", path_img2)
        
        output_img = "/result"+str(n_exec)+".png"
        path_imgi = output_img
        
        #Execute NN
        
        command_execute = "cd "+remote+"\n"
        command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
        command_execute += "export PATH\n"
        command_execute += "source activate sepconv\n"
        command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
        command_execute += "export CUDA_HOME\n"
        command_execute += "CUDA_VISIBLE_DEVICES=1\n"
        command_execute += "export CUDA_VISIBLE_DEVICES\n"
    #    command_execute += "conda info --envs\n"
        command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                            " --second "+path_img2+" --out ./"+path_imgi+"\n"
        command_execute += "rm "+path_img1
        command_execute += "rm "+path_img2
        
        print("Running interpolation of band",band,"out of",d)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
        
        #Get interpolated image
        print("Downloading ireplicated image...")
        sftp.get("./"+path_imgi, "./"+path_imgi)
        imgi = cv2.imread("./"+path_imgi)
        remove("./"+path_imgi)
        new_img.append(imgi[:,:,1])
        #Finish 
        command_execute = "cd "+remote+"\n"
        command_execute+= "rm "+path_imgi
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
    new_img = np.stack(new_img, axis=2)
    ssh.close()
    
    return new_img

def sepconvNN_BOTH(img1, img2, **kwargs):
    """
    Computes SepConvNN for Spectral Imagery with an BOTH window (sliding window,
    one band is computed every time)
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    _,_,deph = img1.shape
    new_img = list()
    for band in range(1,deph-1):
        #Save images before sending them
        bandbf = band-1 if band           else band
        bandnt = band   if band==(deph-1) else band+1

        cv2.imwrite("image1.png", img1[:,:,(bandbf,band,bandnt)])
        cv2.imwrite("image2.png", img2[:,:,(bandbf,band,bandnt)])
        
        #Trasnfer Images
        sftp = ssh.open_sftp()
        sftp.chdir(remote)
        path_img1 = "./images/"+name_img1+str(n_exec)+".png"
        path_img2 = "./images/"+name_img2+str(n_exec)+".png"
        print("Uploading image 1...")
        sftp.put("image1.png", path_img1)
        print("Uploading image 2...")
        sftp.put("image2.png", path_img2)
        
        output_img = "/result"+str(n_exec)+".png"
        path_imgi = output_img
        
        #Execute NN
        
        command_execute = "cd "+remote+"\n"
        command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
        command_execute += "export PATH\n"
        command_execute += "source activate sepconv\n"
        command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
        command_execute += "export CUDA_HOME\n"
        command_execute += "CUDA_VISIBLE_DEVICES=1\n"
        command_execute += "export CUDA_VISIBLE_DEVICES\n"
    #    command_execute += "conda info --envs\n"
        command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                            " --second "+path_img2+" --out ./"+path_imgi+"\n"
        command_execute += "rm "+path_img1
        command_execute += "rm "+path_img2
        
        print("Running interpolation of band",band,"out of",deph)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
        
        #Get interpolated image
        print("Downloading inter_both image...")
        sftp.get("./"+path_imgi, "./"+path_imgi)
        imgi = cv2.imread("./"+path_imgi)
        remove("./"+path_imgi)
        new_img.append(imgi[:,:,1])
        #Finish 
        command_execute = "cd "+remote+"\n"
        command_execute+= "rm "+path_imgi
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
    try:
        new_img = np.stack(new_img, axis=2)
    except:
        new_img = np.stack(new_img, axis=2)
    ssh.close()
    
    return new_img

def sepconvNN_CONS(img1, img2, usable_bands = None, **kwargs):
    """
    Computes SepConvNN for Spectral Imagery with an CONSECUTIVE window
    3-band computation each iteration
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    _,_,deph = img1.shape
    if(usable_bands is None): 
        usable_bands = [x for x in range(deph)]
    new_img = list()
    for x in range(0,int(deph/3)):
        #Save images before sending them
        band0 = x*3
        band1 = band0+1
        band2 = band1+1
        
        cv2.imwrite("image1.png", img1[:,:,(band0,band1,band2)])
        cv2.imwrite("image2.png", img2[:,:,(band0,band1,band2)])
        
        #Trasnfer Images
        sftp = ssh.open_sftp()
        sftp.chdir(remote)
        path_img1 = "./images/"+name_img1+str(n_exec)+".png"
        path_img2 = "./images/"+name_img2+str(n_exec)+".png"
        print("Uploading image 1...")
        sftp.put("image1.png", path_img1)
        print("Uploading image 2...")
        sftp.put("image2.png", path_img2)
        
        output_img = "/result"+str(n_exec)+".png"
        path_imgi = output_img
        
        #Execute NN
        
        command_execute = "cd "+remote+"\n"
        command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
        command_execute += "export PATH\n"
        command_execute += "source activate sepconv\n"
        command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
        command_execute += "export CUDA_HOME\n"
        command_execute += "CUDA_VISIBLE_DEVICES=1\n"
        command_execute += "export CUDA_VISIBLE_DEVICES\n"
    #    command_execute += "conda info --envs\n"
        command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                            " --second "+path_img2+" --out ./"+path_imgi+"\n"
        command_execute += "rm "+path_img1
        command_execute += "rm "+path_img2
        
        print("Running interpolation of band",band2,"out of",deph)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
        
        #Get interpolated image
        print("Downloading inter_both image...")
        sftp.get("./"+path_imgi, "./"+path_imgi)
        imgi = cv2.imread("./"+path_imgi)
        remove("./"+path_imgi)
        if(band0 in usable_bands): new_img.append(imgi[:,:,0])
        if(band1 in usable_bands): new_img.append(imgi[:,:,1])
        if(band2 in usable_bands): new_img.append(imgi[:,:,2])
        #Finish 
        command_execute = "cd "+remote+"\n"
        command_execute+= "rm "+path_imgi
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
    try:
        new_img = np.stack(new_img, axis=2)
    except:
        new_img = np.stack(new_img, axis=2)
    ssh.close()
    
    return new_img

def sepconvNN_JUMP(img1, img2, usable_bands = None, **kwargs):
    """
    Computes SepConvNN for Spectral Imagery with an SPARSED JUMP window
    3-band computation each iteration
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    _,_,deph = img1.shape
    new_img = list()
    if(usable_bands is None): 
        usable_bands = [x for x in range(deph)]
    
    new_img = [None for x in range(len(usable_bands))]
    step = int(deph / 3)
    for x in range(0,int(deph/3)):
        #Save images before sending them
        band0 = x+1
        band1 = band0+step
        band2 = band1+step
        
        cv2.imwrite("image1.png", img1[:,:,(band0,band1,band2)])
        cv2.imwrite("image2.png", img2[:,:,(band0,band1,band2)])
        
        #Trasnfer Images
        sftp = ssh.open_sftp()
        sftp.chdir(remote)
        path_img1 = "./images/"+name_img1+str(n_exec)+".png"
        path_img2 = "./images/"+name_img2+str(n_exec)+".png"
        print("Uploading image 1...")
        sftp.put("image1.png", path_img1)
        print("Uploading image 2...")
        sftp.put("image2.png", path_img2)
        
        output_img = "/result"+str(n_exec)+".png"
        path_imgi = output_img
        
        #Execute NN
        
        command_execute = "cd "+remote+"\n"
        command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
        command_execute += "export PATH\n"
        command_execute += "source activate sepconv\n"
        command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
        command_execute += "export CUDA_HOME\n"
        command_execute += "CUDA_VISIBLE_DEVICES=1\n"
        command_execute += "export CUDA_VISIBLE_DEVICES\n"
    #    command_execute += "conda info --envs\n"
        command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                            " --second "+path_img2+" --out ./"+path_imgi+"\n"
        command_execute += "rm "+path_img1
        command_execute += "rm "+path_img2
        
        print("Running interpolation of band",band2,"out of",deph)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
        
        #Get interpolated image
        print("Downloading inter_both image...")
        sftp.get("./"+path_imgi, "./"+path_imgi)
        imgi = cv2.imread("./"+path_imgi)
        remove("./"+path_imgi)
        if(band0 in usable_bands): new_img[band0-usable_bands[0]] = imgi[:,:,0]
        if(band1 in usable_bands): new_img[band1-usable_bands[0]] = imgi[:,:,1]
        if(band2 in usable_bands): new_img[band2-usable_bands[0]] = imgi[:,:,2]
        #Finish 
        command_execute = "cd "+remote+"\n"
        command_execute+= "rm "+path_imgi
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
    try:
        new_img = np.stack(new_img, axis=2)
    except:
        new_img = np.stack(new_img, axis=2)
    ssh.close()
    
    return new_img

def sepconvNN_TRES(img1, img2, usable_bands = None, **kwargs):
    """
    Computes SepConvNN for Spectral Imagery with an THREE-BAND window
    3-band computation in only one iteration
    """
    connection_path_json = path.dirname(path.realpath(__file__))+".\\connection.json"
    if path.exists(connection_path_json):
        with open(connection_path_json) as f:
            connection = json.load(f)
        server = connection["server"]
        username = connection["user"]
        password = connection["password"]
        port     = connection["port"]
        remote   = connection["remote"]
    else:
        server = None
        username = None
        password = None
        port = None
        remote = None
    
    server = kwargs.get("server", server)
    username=kwargs.get("username",username)
    password=kwargs.get("password",password)
    port    =kwargs.get("port",port)
    remote  =kwargs.get("remote",remote)
    
    name_img1 = kwargs.get("name_img1","first")
    name_img2 = kwargs.get("name_img1","second")
    model     = kwargs.get("modelNN", "lf")
    
    if(server is None or username is None or password is None or port is None or remote is None):
        raise(ValueError("One or more of the required parameters is None:\
                         (server, username, password, port) ->" \
                         + str(server), str(username), str(password), str(port), str(remote)))
    
    n_exec = update_num_proc("metadata")
    
    #Open connection
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password, port=port)
    
    _,_,deph = img1.shape
    new_img = list()
    if(usable_bands is None):
        raise(ValueError("usable_bands has to be specified"))
    elif(len(usable_bands) != 3):
        raise(ValueError("usable_bands has to be 3 numbers exactly"))
    
    new_img = [None for x in range(len(usable_bands))]
    step = int(deph / 3)
    for x in range(0,int(deph/3)):
        #Save images before sending them
        band0 = x
        band1 = band0+step
        band2 = band1+step
        
        cv2.imwrite("image1.png", img1[:,:,(band0,band1,band2)])
        cv2.imwrite("image2.png", img2[:,:,(band0,band1,band2)])
        
        #Trasnfer Images
        sftp = ssh.open_sftp()
        sftp.chdir(remote)
        path_img1 = "./images/"+name_img1+str(n_exec)+".png"
        path_img2 = "./images/"+name_img2+str(n_exec)+".png"
        print("Uploading image 1...")
        sftp.put("image1.png", path_img1)
        print("Uploading image 2...")
        sftp.put("image2.png", path_img2)
        
        output_img = "/result"+str(n_exec)+".png"
        path_imgi = output_img
        
        #Execute NN
        
        command_execute = "cd "+remote+"\n"
        command_execute += "PATH=$PATH:/home/azemar/anaconda2/bin\n"
        command_execute += "export PATH\n"
        command_execute += "source activate sepconv\n"
        command_execute += "CUDA_HOME=/usr/local/cuda-8.0\n"
        command_execute += "export CUDA_HOME\n"
        command_execute += "CUDA_VISIBLE_DEVICES=1\n"
        command_execute += "export CUDA_VISIBLE_DEVICES\n"
    #    command_execute += "conda info --envs\n"
        command_execute += "python run.py --model "+model+ " --first "+path_img1+\
                            " --second "+path_img2+" --out ./"+path_imgi+"\n"
        command_execute += "rm "+path_img1
        command_execute += "rm "+path_img2
        
        print("Running interpolation of band",band2,"out of",deph)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
        
        #Get interpolated image
        print("Downloading inter_both image...")
        sftp.get("./"+path_imgi, "./"+path_imgi)
        imgi = cv2.imread("./"+path_imgi)
        remove("./"+path_imgi)
        if(band0 in usable_bands): new_img[band0-usable_bands[0]] = imgi[:,:,0]
        if(band1 in usable_bands): new_img[band1-usable_bands[0]] = imgi[:,:,1]
        if(band2 in usable_bands): new_img[band2-usable_bands[0]] = imgi[:,:,2]
        #Finish 
        command_execute = "cd "+remote+"\n"
        command_execute+= "rm "+path_imgi
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command_execute)
        ssh_stdout.channel.recv_exit_status()
    try:
        new_img = np.stack(new_img, axis=2)
    except:
        new_img = np.stack(new_img, axis=2)
    ssh.close()
    
    return new_img
