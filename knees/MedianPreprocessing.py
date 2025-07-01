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



iid = ['9011115','9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9073948','9080864','9083500','9089627','9090290','9093622']   
for j in range(0,len(iid)):
    imageName = 'D:/all_kl0_images_masks/knee'+str(iid[j])+'.hdr'
    image = sitk.ReadImage(imageName)
    median_image = median_filter(image,radius=None)
    sitk.WriteImage(median_image, 'D:/median_imgs/median'+str(iid[j])+'.hdr')


    