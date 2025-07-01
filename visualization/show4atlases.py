# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:59:33 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:39:10 2023

@author: ypatia
"""

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

#xalia eikones  3,10,11,15,18,19,31,35,40,41,42

ite='01'
n='4'

#imageName='C:/Users/ypatia/diplomatiki/median_imgs/median'+ite+'.hdr'
imageName2='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
#imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'

#imageName1='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_sbj_111.hdr'
#imageName2='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_sbj_111_brain.nii'

#imageName = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
#imageName = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
#imageName = 'C:/Users/ypatia/diplomatiki/stand_imgs/stand'+ite+'.hdr'
imageName1='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_sbj_111_brain.nii'

#imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_111_t88_masked_gfc_fseg.hdr'
#imageName='C:/Users/jaime/Desktop/normibet/normalized01.hdr'
#args = parser.parse_args()
image1 = sitk.ReadImage(imageName1)
image2 = sitk.ReadImage(imageName2)

#mask = sitk.ReadImage(maskName)

image_arr1 = sitk.GetArrayFromImage(image1)
image_arr2 = sitk.GetArrayFromImage(image2)

image_arr3 = np.swapaxes(image_arr1,0,1)
image_arr3 = np.swapaxes(image_arr3,1,2)

image_arr4 = np.swapaxes(image_arr2,0,1)
image_arr4 = np.swapaxes(image_arr4,1,2)
#mask_arr = sitk.GetArrayFromImage(mask)

#image_arr = np.swapaxes(image_arr,0,1)
#image_arr = np.swapaxes(image_arr,1,2)


ite =[150]
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
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot the first subplot in the upper left position
    axs[0, 0].imshow(image_arr[100,:,:], cmap='gray')
    axs[0, 0].set_title('Original MRI')
    axs[0, 0].axis('off')
    # Plot the second subplot in the upper middle position
    axs[0, 1].imshow(mask_arr[i,:,:], cmap='gray')
    axs[0, 1].set_title('Scull stripped MRI')
    axs[0, 1].axis('off')

    # Plot the third subplot in the lower left position
    axs[1, 0].imshow(image_arr3[i,:,:], cmap='gray')  
    axs[1, 0].axis('off')
    
    # Plot the fourth subplot in the lower middle position
    axs[1, 1].imshow(image_arr4[i,:,:], cmap='gray')
    axs[1, 1].axis('off')
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    