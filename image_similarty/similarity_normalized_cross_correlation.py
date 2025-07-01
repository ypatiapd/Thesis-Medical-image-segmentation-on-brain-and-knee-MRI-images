# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:46:07 2023

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

# Read in the first image



imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']

counter=0
images= list()
dists = list()
image1 = sitk.ReadImage('C:/Users/jaime/Desktop/NormiBets/normi01.hdr')


for ite in imgs:      
    
    dists = list()
    # Read in the second image
    image2 = sitk.ReadImage('C:/Users/jaime/Desktop/NormiBets/normi'+ite+'.hdr')
    
    image1_arr = sitk.GetArrayFromImage(image1)

    # Read in the second image
    image2_arr = sitk.GetArrayFromImage(image2)

    
    mi = normalized_mutual_info_score(image1_arr.ravel(), image2_arr.ravel())

    print("Mutual Information similarity:", mi)
    dists.append(mi)
    dists.append(ite)
    images.append(dists)
    
       

misorted_list = list()
misorted_list = sorted(images, key=lambda x: x[0], reverse = True)

