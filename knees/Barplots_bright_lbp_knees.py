# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:19:52 2023

@author: jaime
"""

import logging
import os

import SimpleITK as sitk
import six
import math
import pandas as pd
import radiomics
from radiomics import featureextractor, getFeatureClasses
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import time 




#ttd ton pinaka dataset kanton normalize ana feature. isws prepei na apothikeftei san array 

def tqdmProgressbar():
  """
  This function will setup the progress bar exposed by the 'tqdm' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality
  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import tqdm
  radiomics.progressReporter = tqdm.tqdm


def clickProgressbar():
  """
  This function will setup the progress bar exposed by the 'click' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality.
  Because the signature used to instantiate a click progress bar is different from what PyRadiomics expects, we need to
  write a simple wrapper class to enable use of a click progress bar. In this case we only need to change the 'desc'
  keyword argument to a 'label' keyword argument.
  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  # Enable the GLCM class to show the progress bar
  extractor.enableFeatureClassByName('glcm')

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import click

  class progressWrapper:
    def __init__(self, iterable, desc=''):
      # For a click progressbar, the description must be provided in the 'label' keyword argument.
      self.bar = click.progressbar(iterable, label=desc)

    def __iter__(self):
      return self.bar.__iter__()  # Redirect to the iter function of the click progressbar

    def __enter__(self):
      return self.bar.__enter__()  # Redirect to the enter function of the click progressbar

    def __exit__(self, exc_type, exc_value, tb):
      return self.bar.__exit__(exc_type, exc_value, tb)  # Redirect to the exit function of the click progressbar

  radiomics.progressReporter = progressWrapper

import numpy as np
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

import dipy 
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
#7,15,16,20,26,34,38,39 n3
#imgs=['14','17','18','19','21','22','23','25','27','28','29','30','31','32','33','35','37','40','41','42']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','15','16','20','26','34','38','39']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','3','3','3','3','3','3','3']
imgs=['01']
#imgs=['26']#,'38','09']
#n=['3','3','4','3',
n=['4']#,'3','4']
#imgs=['01','11']
#n=['4','4']

counter=0

for ite in imgs:   

    #imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr'
    imageName='D:/norm_imgs/norm9011115.hdr'
    #imageName='D:/registered_imgs/9011115/mri.hdr'
    #imageName='D:/median_imgs/median9011115.hdr'

    #imageName='D:/hist_imgs/hist9019287.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
    #maskName = 'D:/all_kl0_images_masks/mask9011115.hdr'
    maskName  = 'D:/registered_imgs/9011115/mask.hdr'
    image= sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    mask_arr = sitk.GetArrayFromImage(mask)
    #mask_arr = np.flip(mask_arr,0)
    
    
    mask = sitk.GetImageFromArray(mask_arr)
    #maskName='C:/Users/ypatia/diplomatiki/registered_masks/registered_'+ite+'.hdr'
    #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    
    
    a=sitk.RegionOfInterestImageFilter()
    a.SetRegionOfInterest([15,74,101,116,203,157]) # slack ROI for all Images in imgs : [106, 79, 20, 253, 272, 126]    
    mask=a.Execute(mask)
    image=a.Execute(image) 
    
    
    #I = radiomics.imageoperations.getLoGImage(image, mask, **kwargs)
    #img = next(I)
    #image = img[0]
    #image = sitk.GetImageFromArray(imag)
    
    #L=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2)

    applyLog = False
    applyWavelet = False
    
    # Setting for the feature calculation.
    # Currently, resampling is disabled.
    # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
    
    # Regulate verbosity with radiomics.verbosity
    # radiomics.setVerbosity(logging.INFO)
    
    # Get the PyRadiomics logger (default log-level = INFO
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
    
    # Write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    
    pdf = pd.DataFrame()
    
    
    pltcnt = 0
    dataset = list()
    dataset2 = list()
    dataset3 = list()
    dataset4 = list()

    imgdim = sitk.GetArrayFromImage(image)
    dim1 = len(imgdim)
    dim2 = len(imgdim[0])
    dim3 = len(imgdim[0][0])
    
    
    mask_view = sitk.GetArrayFromImage(mask)
    bright_view = sitk.GetArrayFromImage(image)
    #if z<len(featlist)-1:
    '''
    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=3)
    
    c1 = next(L1)
    level1= c1[0]   
    b1 =next(L1)
    level2= b1[0]
    lbp_arr1 = sitk.GetArrayFromImage(level1)    
    c2 = next(L2)
    level21= c2[0]
    b2 =next(L2)
    level22= b2[0] 
    lbp_arr21 = sitk.GetArrayFromImage(level21)
    c3 = next(L3)
    level31= c3[0]
    b3 =next(L3)
    level32= b3[0] 
    lbp_arr31 = sitk.GetArrayFromImage(level31)
    
    
    lbp_arr = np.zeros((len(lbp_arr1), len(lbp_arr1[0]), len(lbp_arr1[0][0])))
    for i in range(0,len(lbp_arr1)):
        for j in range(0,len(lbp_arr1[0])):
            for k in range(0,len(lbp_arr1[0][0])):
                
                if  lbp_arr31[i][j][k]!=0 and lbp_arr21[i][j][k]!=0 :
                    print('yes')
                if lbp_arr1[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr1[i][j][k]
                elif lbp_arr21[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr21[i][j][k]
                elif lbp_arr31[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr31[i][j][k] 
                
     ''' 
    
    counter = 0
    for i in range(0,dim1):
        for j in range(0,dim2):
            for k in range(0,dim3):
                temp = list()
                temp.append(mask_view[i][j][k]) #pairnoume ti maska gia na skiparoume mideniki klasi sto dataset
                temp.append(bright_view[i][j][k])
                
                #temp.append(lbp_arr[i][j][k])
                dataset.append(temp)                
                #temp.append(bright_view[i][j][k])
                #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                #temp.append(dist)
                '''
                if mask_view[i][j][k] != 0 and bright_view[i][j][k] != 0: #skipparoume mideniki klasi
                    dataset.append(temp)
                
                if mask_view[i][j][k] != 0 and bright_view[i][j][k] == 0: #skipparoume mideniki klasi
                    dataset.append(temp)
                if mask_view[i][j][k] == 0 and bright_view[i][j][k] != 0: #skipparoume mideniki klasi
                    dataset2.append(temp)
                if mask_view[i][j][k] != 0 and bright_view[i][j][k] != 0: #skipparoume mideniki klasi
                    dataset3.append(temp)
                if mask_view[i][j][k] == 0 and bright_view[i][j][k] == 0: #skipparoume mideniki klasi
                    dataset4.append(temp)
                '''
                    
    print(len(dataset))
    print(len(dataset2))                
    print(len(dataset3))
    print(len(dataset4))
    
    class1 = list()
    class2 = list()
    class3 = list()
    class4 = list()
    for d in range(0,len(dataset)):
    #for q in range(0,1): 
         
         pltcnt += 1
         #for d in range(0,len(dataset)-1):
         if dataset[d][0] == 1:
             class1.append(dataset[d][1])
         elif dataset[d][0] == 2:    
             class2.append(dataset[d][1])
         elif dataset[d][0] == 3:      
             class3.append(dataset[d][1])
         elif dataset[d][0] == 4:
             class4.append(dataset[d][1])
         
        
    plt.figure(1)
    plt.hist(class1,bins=100,color='blue')
    '''
    plt.figure(2)
    counts, bins = np.histogram(class1)
    plt.stairs(counts, bins)
    '''
   
    plt.figure(2)
    plt.hist(class2,bins=100,color='red')
    plt.figure(3)
    plt.hist(class3,bins=100,color='green')
    
    plt.figure(4)
    plt.hist(class4,bins=100,color='yellow')
    
    
    
    class0 = list()
    class0.append(class1)
    class0.append(class2)
    class0.append(class3)
    class0.append(class4)
    plt.figure(5)             
    colors = ['green', 'blue', 'lime','purple']
    labels = ['1','2','3','4']
    plt.hist(class0,bins=100,color=colors,label=labels,density=True)
    plt.legend(prop ={'size': 10})
     
    plt.title('brightness 3erwgw''\n\n',
             fontweight ="bold")
    #plt.title('matplotlib.pyplot.hist() function Example\n\n',fontweight ="bold")
     
    plt.show()


    