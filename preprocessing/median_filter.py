# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:02:46 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:57:55 2023

@author: jaime
"""

'''
Image Preprocessing Utilities

1. Bias Field Correction

2. Denoising 

3. Sharpening

4. Intensity Standardization

'''

import os
import tempfile
import shutil
import gc

import math

import numpy as np
import SimpleITK as sitk

from scipy.interpolate import interp1d
from sklearn.utils.extmath import cartesian
from skimage.exposure import rescale_intensity

#from ants import ants_image_io, ants_image
#from ants.utils import n3_bias_field_correction

from dipy.denoise import noise_estimate, nlmeans

#from numba import jit, prange
from joblib import Parallel, delayed, dump, load



# ---------------- #
# Median Filtering #
# ---------------- #
def median_filter(image, radius = None):
    '''
    Median filter
    
    '''
    med_filt = sitk.MedianImageFilter()

    if radius is not None:

        if hasattr(radius, '__len__'):
            assert len(radius) == 3, 'Incorrect number of elements'
        med_filt.SetRadius(list(radius))

    else:

        image = med_filt.Execute(image)

    return image


imgs=['01']#,'02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
n=['4']#,'4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
counter=0

for ite in imgs:
    imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'

    image = sitk.ReadImage(imageName)  
    median_image = median_filter(image,radius=None)   
    sitk.WriteImage(median_image, 'C:/Users/ypatia/diplomatiki/median_imgs/median'+ite+'.hdr')
    counter=counter+1