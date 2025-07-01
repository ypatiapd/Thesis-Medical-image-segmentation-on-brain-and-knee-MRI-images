# -*- coding: utf-8 -*-
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

#imageName = 'C:/Users/jaime/YanAlgorithm/denoised_imgs/denoised01.hdr'

#xalia eikones  3,10,11,15,18,19,35,40,41,42
#19,40,42 kali
ite='26'
n='3'

#imageName='C:/Users/ypatia/diplomatiki/median_imgs/median'+ite+'.hdr'
#imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
#imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'

#imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_sbj_111_brain.nii'
imageName = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
#imageName = 'D:/disc1/OAS1_0002_MR1/FSL_SEG/OAS1_0002_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
#imageName = 'C:/Users/ypatia/diplomatiki/stand_imgs/stand'+ite+'.hdr'

#imageName='D:/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
#maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_111_t88_masked_gfc_fseg.hdr'
maskName= 'D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_111_t88_masked_gfc_fseg.hdr'
#imageName='C:/Users/jaime/Desktop/normibet/normalized01.hdr'
#args = parser.parse_args()
image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)

image_arr = sitk.GetArrayFromImage(image)
mask_arr = sitk.GetArrayFromImage(mask)


#image_arr = np.swapaxes(image_arr,0,1)
#image_arr = np.swapaxes(image_arr,1,2)

'''
image=sitk.GetImageFromArray(image_arr)

# Get the mask and image arrays
mask_arr = sitk.GetArrayFromImage(mask)
image_arr = sitk.GetArrayFromImage(image)

# Print the shape of the image array
print(image_arr.shape)


# Create a figure with two subplots
dim1, dim2, dim3 = image_arr.shape

# Create a figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the image
ax.imshow(image_arr[146,:,:], cmap='gray')

# Show the figure
plt.show()


dim1, dim2, dim3 = image_arr.shape

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the mask in the first subplot
ax1.imshow(mask_arr[146,:,:], cmap='gray')
ax1.set_title('Mask')

# Plot the image in the second subplot
ax2.imshow(image_arr[146,:,:], cmap='gray')
ax2.set_title('Image')

# Show the figure
plt.show()
'''

ite =[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,159]
#ite=[90]
for i in ite:
    '''
    dim1, dim2, dim3 = image_arr.shape
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the image
    ax.imshow(image_arr[i,:,:], cmap='gray')
    
    # Show the figure
    plt.show()
    '''
    
    dim1, dim2, dim3 = image_arr.shape
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the mask in the first subplot
    ax1.imshow(image_arr[:,:,i], cmap='gray')
    ax1.set_title('Registered image')
    
    # Plot the image in the second subplot
    ax2.imshow(mask_arr[:,:,i], cmap='gray')
    ax2.set_title('Mask')
