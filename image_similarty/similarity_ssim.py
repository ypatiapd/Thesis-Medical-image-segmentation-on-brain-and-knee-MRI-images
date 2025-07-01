# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:17:30 2023

@author: jaime
"""

import SimpleITK
import SimpleITK as sitk
import numpy as np
import numpy as np
from scipy.ndimage import correlate
from numpy.fft import fft2, ifft2, fftshift
#from skimage.measure import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim


# Read in the first image



imgs=['01','02','04','05','06','07','09','12','13','14','16','17','20','21','22','23','25','27','28','33','34','37','38','39']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']

counter=0
images= list()
dists = list()
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')


for ite in imgs:      
    
    dists = list()
    # Read in the second image
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    
    image1_arr = sitk.GetArrayFromImage(image1)

    # Read in the second image
    image2_arr = sitk.GetArrayFromImage(image2)

    
    ssim_value = ssim(image1_arr, image2_arr, multichannel=True)

    print("Structural Similarity Index (SSIM) similarity:", ssim_value)
    dists.append(ssim_value)
    dists.append(ite)
    images.append(dists)
    
       

ssimsorted_list = list()
ssimsorted_list = sorted(images, key=lambda x: x[0], reverse = True)

