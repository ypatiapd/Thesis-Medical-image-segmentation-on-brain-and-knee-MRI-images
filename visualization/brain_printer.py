# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:48:46 2022

@author: ypatia
"""

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
imgs=['01']
n=['4']
#imgs=['38','25','05','26','07','29','06','39','09']
#n=['3','4','4','3','3','4','4','3','4']

#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['01',]
#n=['4']
counter=0;
for ite in imgs:   
    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    maskName='D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
    imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'

    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+z+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    image = sitk.ReadImage(imageName)
    
    mask = sitk.ReadImage(maskName)
    
    
 
    
    
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
            print(image_arr[q][i][j])
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
    
    label_view = sitk.GetArrayFromImage(mask)
    bright_view = sitk.GetArrayFromImage(image)
    for i in range(0,len(bright_view)):
        for j in range(0,len(bright_view[0])):
            for k in range(0,len(bright_view[0][0])):
                if bright_view[i][j][k]!=0 and label_view[i][j][k]==1:
                    x1.append(bright_view[i][j][k])
                elif bright_view[i][j][k]!=0 and label_view[i][j][k]==2:
                    x2.append(bright_view[i][j][k])
                elif bright_view[i][j][k]!=0 and label_view[i][j][k]==3:
                    x3.append(bright_view[i][j][k])
                    
    plt.figure(100)
    plt.hist(x1,bins=100,color='blue')
    plt.hist(x2,bins=100,color='green')
    plt.hist(x3,bins=100,color='red')
    plt.show()