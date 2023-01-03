#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:05:31 2022
@author: carlottatrigila
"""

import numpy as np
import os,glob
from   os import path
import matplotlib.pyplot as plt

pathImages =  "/Users/kajewys/Workspace/Roncali/TrainingImages/full_set/"

Directory  = os.listdir(pathImages)
dat        = []
label      = []
Image_i    = []
label_i    = 0

for g in range(0,len(Directory)):
    pathway = pathImages+str(Directory[g])
    ########################################################################
    ######### assign a label to the figure
    ########################################################################

    if "0_0_1.5" in Directory[g]:
        label_i= 0;
    if "0_0_2.5" in Directory[g]:
        label_i= 1;
    if "075_075_1.5" in Directory[g]:
        label_i= 2;
    if "075_075_2.5" in Directory[g]:
        label_i= 3;

    
    ########################################################################
    ########################################################################
    Image_i = np.load(pathway)
    ###### rescale from 0 to 255 as in Fashion MNIST
    Image_i= np.round(np.interp(Image_i, (Image_i.min(), Image_i.max()), (0, 255)))
    
    ###### Plot the image
    # plt.imshow(Image_i, cmap='gray_r')
    # plt.title(str(label_i))
    # plt.show()
    # plt.close()
    ######
    
    Image_i = Image_i.astype('float32')

    #pyplot.imshow(Image_i, cmap='gray_r')
    #pyplot.show()
    if "0_0_2.5" in Directory[g]:
        print(g)
        if dat==[]:
            dat   = [Image_i]
            label = [label_i]
        else:
            dat.append(Image_i)
            label.append(label_i)

Dataset=np.array(dat)
Labels =np.array(label); Labels = Labels.astype('float32')

#### shuffle all the data 
#Labels_rnd,Dataset_rnd= shuffle(Labels,Dataset)

del label_i, Image_i, g, Directory, dat, label

np.save("/Users/kajewys/Workspace/Roncali/TrainingImages/Labels.npy", Labels)
#np.save( "/Users/kajewys/Workspace/Roncali/TrainingImages/Dataset.npy", Dataset)
np.save( "/Users/kajewys/Workspace/Roncali/TrainingImages/Dataset_center.npy", Dataset)