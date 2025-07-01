# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:57:19 2023

@author: ypatia
"""

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

#7,15,16,20,26,34,38,39 n3
#imgs=['13','28','33','23','22','16','11','20']
#n=['3','3','4','3',
#n=['4','4','4','4','4','3','4','3']
#imgs=['38','25','05','26','07','29','06','39','09']
#n=['3','4','4','3','3','4','4','3','4']
imgs=['01','02','04','05','06','07','09','11','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','31','32','33','34','37','38','39','40','42']
n=['4','4','4','4','4','3','4','4','4','4','4','3','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','3','3','4','4']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['11']#,'02','04','05','06','07']
#n=['4']#,'4','4','4','4','3']
counter=0;
for ite in imgs:    
    imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'
    #maskName= 'D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    # Read the input image
    input_image = sitk.ReadImage(imageName)
    # Convert the input image to a float image
    input_image_float = sitk.Cast(input_image, sitk.sitkFloat32)

    # Compute the mean and standard deviation of the image intensity values
    stat_filter = sitk.StatisticsImageFilter()
    stat_filter.Execute(input_image_float)
    image_mean = stat_filter.GetMean()
    image_stddev = stat_filter.GetSigma()
    
    # Compute the z-score normalized image
    zscore_filter = sitk.ShiftScaleImageFilter()
    zscore_filter.SetShift(-1.0 * image_mean)
    zscore_filter.SetScale(1.0 / image_stddev)
    output_image = zscore_filter.Execute(input_image_float)
    # Write the z-score normalized image to disk
   
    sitk.WriteImage(output_image, 'C:/Users/ypatia/diplomatiki/zscore_imgs/zscore'+ite+'.hdr')
    
    counter+=1