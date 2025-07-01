# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:39:53 2023

@author: jaime
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt


'''

image = sitk.ReadImage('C:/Users/jaime/Downloads/MRI_Data/OAI-ZIB/segmentation_masks/9001104.segmentation_masks.mhd')
image_array = sitk.GetArrayFromImage(image)
plt.imshow(image_array, cmap='gray')
plt.show()
'''
"""
Created on Tue Dec 20 18:59:58 2022

@author: jaime
"""
import logging
import os

import SimpleITK as sitk
import six
import math
import pandas as pd
#import radiomics
#from radiomics import featureextractor, getFeatureClasses
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import time 
#import platipy
#from platipy.imaging.registration.utils import apply_transform

import numpy as np
import six

#from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

import dipy 
import warnings

import argparse
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk


import matplotlib.pyplot as plt
import pydicom

full_mhd_path = 'C:/Users/jaime/Downloads/MRI_Data/OAI-ZIB/segmentation_masks/9001104.segmentation_masks.mhd'

mask = sitk.ReadImage(full_mhd_path)
mask_arr = sitk.GetArrayFromImage(mask)
mask_arr = np.flip(mask_arr,0)


mask = sitk.GetImageFromArray(mask_arr)

s = (len(mask_arr),len(mask_arr[0]))
total = np.ones(s)

imageName='C:/Users/jaime/Desktop/KneeImages/knee9001104.hdr'
image = sitk.ReadImage(imageName)
image_arr = sitk.GetArrayFromImage(image)


for i in range(1,16):        
    
    k = i * 10
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the mask in the first subplot
    ax1.imshow(mask_arr[:,:,k-1], cmap='gray')
    ax1.set_title('Mask')
    
    # Plot the image in the second subplot
    ax2.imshow(image_arr[:,:,k-1], cmap='gray')
    ax2.set_title('Image slice '+str(k)+'')


