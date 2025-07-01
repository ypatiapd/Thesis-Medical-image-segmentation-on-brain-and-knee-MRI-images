# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:28:45 2023

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

#imageName='C:/Users/ypatia/diplomatiki/median_imgs/median'+ite+'.hdr'
#imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
#imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'

#imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_sbj_111_brain.nii'
'''imageName1 = 'C:/Users/ypatia/diplomatiki/denoised/denoised04.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/denoised/denoised05.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/denoised/denoised06.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/denoised/denoised07.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/denoised/denoised09.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/denoised/denoised12.hdr'
'''

'''
imageName1 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm04.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm29.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm07.hdr'

imageName4 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm12.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm20.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm26.hdr'
'''


'''imageName1 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm02.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm13.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm14.hdr'

imageName4 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm16.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm17.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm28.hdr'
'''


imageName1 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm09.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm12.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm13.hdr'

imageName4 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm16.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm17.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm19.hdr'

'''imageName1 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered06.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered09.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered21.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered22.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered20.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered27.hdr'
'''
'''
imageName1 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered02.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered13.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered14.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered16.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered17.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered28.hdr'
'''
'''imageName1 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered04.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered29.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered27.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered12.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered20.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/registered_imgs/registered26.hdr'
'''

imageName1 = 'C:/Users/ypatia/diplomatiki/denoised/denoised04.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/denoised/denoised29.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/denoised/denoised07.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/denoised/denoised12.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/denoised/denoised20.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/denoised/denoised26.hdr'

'''
imageName1 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist04.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist29.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist07.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist12.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist20.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist26.hdr'
'''
'''
imageName1 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore01.hdr'
imageName2 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore02.hdr'
imageName3 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore04.hdr'
imageName4 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore05.hdr'
imageName5 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore06.hdr'
imageName6 = 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore11.hdr'
'''

'''imageName1 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111_brain.nii'
imageName2 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0002_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0002_MR1_mpr_n4_anon_sbj_111_brain.nii'
imageName3 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0004_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0004_MR1_mpr_n4_anon_sbj_111_brain.nii'
imageName4 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0005_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0005_MR1_mpr_n4_anon_sbj_111_brain.nii'
imageName5 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0007_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0007_MR1_mpr_n3_anon_sbj_111_brain.nii'
imageName6 = 'C:/Users/ypatia/diplomatiki/disc1/OAS1_0009_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0009_MR1_mpr_n4_anon_sbj_111_brain.nii'
'''
#imageName = 'C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
#imageName = 'C:/Users/ypatia/diplomatiki/stand_imgs/stand'+ite+'.hdr'

#imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
#maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n+'_anon_111_t88_masked_gfc_fseg.hdr'
#imageName='C:/Users/jaime/Desktop/normibet/normalized01.hdr'
#args = parser.parse_args()
image1 = sitk.ReadImage(imageName1)
image2 = sitk.ReadImage(imageName2)
image3 = sitk.ReadImage(imageName3)
image4 = sitk.ReadImage(imageName4)
image5 = sitk.ReadImage(imageName5)
image6 = sitk.ReadImage(imageName6)

#mask = sitk.ReadImage(maskName)

image_arr1 = sitk.GetArrayFromImage(image1)
image_arr2 = sitk.GetArrayFromImage(image2)
image_arr3 = sitk.GetArrayFromImage(image3)
image_arr4 = sitk.GetArrayFromImage(image4)
image_arr5 = sitk.GetArrayFromImage(image5)
image_arr6 = sitk.GetArrayFromImage(image6)
#mask_arr = sitk.GetArrayFromImage(mask)

#image_arr = np.swapaxes(image_arr,0,1)
#image_arr = np.swapaxes(image_arr,1,2)

dim1 = len(image_arr1)
dim2 = len(image_arr1[0])
dim3 = len(image_arr1[0][0])
print(dim1)
print(dim2)
print(dim3)



ite =[140]
for i in ite:
    # Create a figure with six subplots
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    
    # Plot the first subplot in the upper left position
    axs[0, 0].imshow(image_arr1[:,:,i], cmap='gray')
    axs[0, 0].set_title('Denoised MRI #1')
    axs[0, 0].axis('off')
    # Plot the second subplot in the upper middle position
    axs[0, 1].imshow(image_arr2[:,:,i], cmap='gray')
    axs[0, 1].set_title('Denoised MRI #2')
    axs[0, 1].axis('off')
    # Plot the third subplot in the upper right position
    axs[0, 2].imshow(image_arr3[:,:,i], cmap='gray')
    axs[0, 2].set_title('Denoised MRI #3')
    axs[0, 2].axis('off')
    # Plot the fourth subplot in the lower left position
    axs[1, 0].imshow(image_arr4[:,:,i], cmap='gray')
    axs[1, 0].set_title('Normalized MRI #1')
    axs[1, 0].axis('off')
    # Plot the fifth subplot in the lower middle position
    axs[1, 1].imshow(image_arr5[:,:,i], cmap='gray')
    axs[1, 1].set_title('Normalized MRI #2')
    axs[1, 1].axis('off')
    # Plot the sixth subplot in the lower right position
    axs[1, 2].imshow(image_arr6[:,:,i], cmap='gray')
    axs[1, 2].set_title('Normalized MRI #3')
    axs[1, 2].axis('off')
    # Adjust the spacing between the subplots
    fig.subplots_adjust(wspace=0.1, hspace=0.001)
    

    
