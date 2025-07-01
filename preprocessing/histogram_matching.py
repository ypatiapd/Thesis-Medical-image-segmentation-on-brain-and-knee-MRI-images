# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:27:12 2023

@author: ypatia
"""




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


def histogram_matching(image, template, n_points = 5, n_levels = 100):
    '''
    Histogram matching image filter

    '''

    assert type(image) == sitk.SimpleITK.Image and type(template) == sitk.SimpleITK.Image, 'Incorrect image type'

    hist_match = sitk.HistogramMatchingImageFilter()
    hist_match.SetNumberOfMatchPoints(n_points)
    hist_match.SetNumberOfHistogramLevels(n_levels)

    image = hist_match.Execute(image, template)

    return image

imgs=['01','02','04','05','06','07','09','11','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','31','32','33','34','37','38','39','40','42']
n=['4','4','4','4','4','3','4','4','4','4','4','3','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','3','3','4','4']
#imgs=['01']
#n=['4']
#templateName = 'C:/Users/ypatia/diplomatiki/norm_imgs/norm23.hdr'
templateName = 'C:/Users/ypatia/diplomatiki/denoised/denoised01.hdr'

template = sitk.ReadImage(templateName)

for ite in imgs:
    #imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'

    image = sitk.ReadImage(imageName)
    
    hist_matched_image = histogram_matching(image, template)
    sitk.WriteImage(hist_matched_image, 'C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
)
