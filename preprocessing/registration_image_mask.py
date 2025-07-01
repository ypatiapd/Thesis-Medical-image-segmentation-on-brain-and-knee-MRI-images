# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:18:31 2022

@author: jaime
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:27:48 2022

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
imgs=['01','02','04','05','06','07','09','11','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','31','32','33','34','37','38','39','40','42']
n=['4','4','4','4','4','3','4','4','4','4','4','3','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','3','3','4','4']
imgs=['12']
n=['4']
#n=['4','4','4','4','4','3','4','3']
#imgs=['38','25','05','26','07','29','06','39','09']
#n=['3','4','4','3','3','4','4','3','4']


counter=0;
for ite in imgs:  
    
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'
    imageName= 'D:/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'

    #imageName='C:/Users/ypatia/diplomatiki/no_scull_imgs/noscull'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'_brain.nii'

    #imageName='C:/Users/ypatia/diplomatiki/median_imgs/median'+ite+'.hdr'

    #imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'

    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    maskName= 'D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'

    #reMask = 'C:/Users/jaime/YanAlgorithm/resegmented_imgs/rsg01.hdr'
    
    
    
    #args = parser.parse_args()
    image = sitk.ReadImage(imageName)
    image_arr = sitk.GetArrayFromImage(image)
    
    
    mask = sitk.ReadImage(maskName)
    
    image_arr = np.swapaxes(image_arr,0,1)
    image_arr = np.swapaxes(image_arr,1,2)
    
    dim1 = len(image_arr)
    dim2 = len(image_arr[0])
    dim3 = len(image_arr[0][0])
    print(dim1)
    print(dim2)
    print(dim3)
    
    
    image=sitk.GetImageFromArray(image_arr)
    
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    dim11 = len(mask_arr)
    dim22 = len(mask_arr[0])
    dim33 = len(mask_arr[0][0])
    
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask)
    
    bbox = np.array(lsif.GetBoundingBox(1))
    bbox2 = np.array(lsif.GetBoundingBox(2))
    bbox3 = np.array(lsif.GetBoundingBox(3))
    
    print(dim11)
    print(dim22)
    print(dim33)
    print('spacing: ' + str(mask.GetSpacing()))
    print('spacing: ' + str(image.GetSpacing()))
    print('number of pixel components: ' + str(image.GetNumberOfComponentsPerPixel()))
    
    
    
    # Create an Elastix filter
    elastix = sitk.ElastixImageFilter()
    
    # Set the fixed image to the mask
    elastix.SetFixedImage(mask)
    
    # Set the moving image to the MRI scan
    elastix.SetMovingImage(image)
    
    # Set the parameters for the registration
    parameter_map = sitk.GetDefaultParameterMap('affine')
    #parameter_map['Interpolator'] = ['NearestNeighborInterpolator']
    elastix.SetParameterMap(parameter_map)
    
    # Run the registration
    elastix.Execute()
    image=elastix.GetResultImage()
    
    sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr')
    # Get the transformation from the registration
    
    
    '''
    transform = elastix.GetTransform()
    
    
    
    
    
    # Create a resampling filter
    resampler = sitk.ResampleImageFilter()
    
    # Set the reference image to the mask
    resampler.SetReferenceImage(mask)
    
    # Set the size of the output image to the size of the mask
    resampled_size = mask.GetSize()
    resampler.SetSize(resampled_size)
    
    # Set the interpolator to linear
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # Set the default pixel value to 0
    resampler.SetDefaultPixelValue(0)
    
    # Set the output spacing to the spacing of the mask
    resampled_spacing = mask.GetSpacing()
    resampler.SetOutputSpacing(resampled_spacing)
    
    # Set the transform
    resampler.SetTransform(transform)
    
    # Resample the image
    resampled_image = resampler.Execute(image)
    
    '''
    
    
    
    
    #outlabsitk = resampler.Execute(sitklabel)
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    
    x1=list()
    y1=list()
    x2=list()
    y2=list()
    x3=list()
    y3=list()
    x4=list()
    x5=list()
    y4=list()
    y5=list()
    x6=list()
    y6=list()
    x7=list()
    y7=list()
    x8=list()
    y8=list()
    x9=list()
    y9=list()
    x10=list()
    y10=list()
    x11=list()
    y11=list()
    #edw plotaroume katopsi
    
    dim1 = len(image_arr)
    dim2 = len(image_arr[0])
    dim3 = len(image_arr[0][0])
    q = round(dim1/2)
    q = 100
    for i in range(0,dim2):
        for j in range(0,dim3):   
            if image_arr[q][i][j]!=0 and image_arr[q][i][j] <= 500:# and mask_arr[i][65][j] != 0 :
                x4.append(j)
                y4.append(i)
            elif image_arr[q][i][j]<=850 and image_arr[q][i][j] > 500:# and mask_arr[i][65][j] != 0 :
                x5.append(j)
                y5.append(i)
            elif image_arr[q][i][j] > 850 :#and image_arr[i][130][j]<=1300 :#and mask_arr[i][65][j] != 0:# and mask_arr[i][65][j] == 0 :
                x10.append(j)
                y10.append(i)
            elif image_arr[q][i][j] == 0 :#and image_arr[i][130][j]<=1300 :#and mask_arr[i][65][j] != 0:# and mask_arr[i][65][j] == 0 :
               x11.append(j)
               y11.append(i) 
            
    arrayx4=np.asarray(x4)
    arrayy4=np.asarray(y4)
    
    arrayx5=np.asarray(x5)
    arrayy5=np.asarray(y5)
    
    arrayx10=np.asarray(x10)
    arrayy10=np.asarray(y10)
    
    arrayx11=np.asarray(x11)
    arrayy11=np.asarray(y11)
    
    plt.scatter(arrayx10, arrayy10,color='green')
    plt.scatter(arrayx11, arrayy11,color='purple')
    
    plt.scatter(arrayx4, arrayy4,color='red')
    plt.scatter(arrayx5, arrayy5,color='blue')
    
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show()
    dim1 = len(mask_arr)
    dim2 = len(mask_arr[0])
    dim3 = len(mask_arr[0][0])
    q= round(dim1/2)
    q = 100
    for i in range(0,dim2):
        for j in range(0,dim3):   #145 171 137
            if mask_arr[q][i][j]==1:# and image_arr[i][j][68]==0 :
                x1.append(j)
                y1.append(i)
            elif mask_arr[q][i][j]==2:# and image_arr[i][j][68]==0:
                x2.append(j)
                y2.append(i)
            elif mask_arr[q][i][j]==3:# and image_arr[i][j][68]==0:
                x3.append(j)
                y3.append(i)
            elif mask_arr[q][i][j]==0:# and image_arr[i][j][68]==0:
                x9.append(j)
                y9.append(i)
              
                
    arrayx1=np.asarray(x1)
    arrayy1=np.asarray(y1)
    arrayx2=np.asarray(x2)
    arrayy2=np.asarray(y2)
    arrayx3=np.asarray(x3)
    arrayy3=np.asarray(y3)
    arrayx9=np.asarray(x9)
    arrayy9=np.asarray(y9)
    
    plt.scatter(arrayx9, arrayy9,color='black')
    
    plt.scatter(arrayx1, arrayy1,color='red')
    plt.scatter(arrayx2, arrayy2,color='blue')
    plt.scatter(arrayx3, arrayy3,color='green')
    #plt.scatter(arrayx9, arrayy9,color='black')
    #plt.scatter(arrayx3, arrayy3,color='yellow')
    
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show()
    counter =counter+1
    '''
    c=0
    for i in range(0,dim1):
        for j in range(0,dim2):   
            for k in range(0,dim3):  
                if image_arr[i][j][k]==0 and mask_arr[i][j][k] != 0 :
                    c+=1
    
    print(c)
    '''