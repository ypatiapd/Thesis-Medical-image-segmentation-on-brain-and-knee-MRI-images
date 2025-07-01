# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:23:00 2023

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
import numpy as np
import matplotlib.pyplot as plt


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

import numpy 
import SimpleITK as sitk
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
#imgs=['38','25','05','26','07','29','06','39','09','13']
#n=['3','4','4','3','3','4','4','3','4']
imgs=['12']
n=['4']

counter=0

for ite in imgs:   
    imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised_and_norm_imgs/norm'+ite+'.hdr'
    maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'

    #maskName='C:/Users/ypatia/diplomatiki/registered_masks/registered_'+ite+'.hdr'
    #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    paramsFile = 'C:/Users/ypatia/diplomatiki/params-1.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    a=sitk.RegionOfInterestImageFilter()
    
    #a.SetRegionOfInterest([90,68,60,10,10,10])
    #a.SetRegionOfInterest([60,102,60,30,34,30])
    #a.SetRegionOfInterest([66,130,88,22,26,22])
    #a.SetRegionOfInterest([90,68,60,30,34,30])
    #a.SetRegionOfInterest([60,68,90,30,34,30])
    #a.SetRegionOfInterest([88,104,132,10,10,10])
    #a.SetRegionOfInterest([88,104,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([60,72,60,88,104,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    
    a.SetRegionOfInterest([44,52,44,88,104,88])
    
    mask=a.Execute(mask)
    image=a.Execute(image) 

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
    
    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    featureClasses = getFeatureClasses()
    
    # Uncomment one of these functions to show how PyRadiomics can use the 'tqdm' or 'click' package to report progress when
    # running in full python mode. Assumes the respective package is installed (not included in the requirements)
    
    tqdmProgressbar()
    # clickProgressbar()
    
    print("Active features:")
    for cls, features in six.iteritems(extractor.enabledFeatures):
      if features is None or len(features) == 0:
        features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
      for f in features:
        print(f)
        print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)
    
    print("Calculating features")
    
    #all_features=numpy.zeros((20,20,20))
    
    
    featureVector = extractor.execute(image, mask,label=1, voxelBased=True)
    featureVector2 = extractor.execute(image, mask,label=2, voxelBased=True)
    featureVector3 = extractor.execute(image, mask,label=3, voxelBased=True)
    
    #kala feat glcm : Autocorrelation,jointAverage
    #kala feat glrlm:  LongRunHighGrayLevelEmphasis,HighGrayLevelRunEmphasis,ShortRunHighGrayLevelEmphasis,
    #kala gldm : grayLevelVariance(kaloutsiko),HighGrayLevelEmphasis
   
    featlist = list()
    featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Kurtosis','original_firstorder_Maximum','original_firstorder_Mean','original_firstorder_MeanAbsoluteDeviation','original_firstorder_Median','original_firstorder_Minimum','original_firstorder_Minimum','original_firstorder_Range','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_TotalEnergy','original_firstorder_Uniformity','original_firstorder_Variance']
    #featlist_glcm = ['original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_Correlation','original_glcm_MaximumProbability','original_glcm_Id','original_glcm_DifferenceAverage'] 
    featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_JointAverage']#,'original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 

    featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis']#,'original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
    
    featlist_gldm = ['original_gldm_GrayLevelVariance','original_gldm_HighGrayLevelEmphasis']#,'original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    #featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    #featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized']#,'original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis']#'original_glszm_LargeAreaLowGrayLevelEmphasis','log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis']
    #featlist_shape= ['original_shape_MeshVolume','original_shape_VoxelVolume']
    
    
        
    featlist.append(featlist_firstorder)
    #featlist.append(featlist_shape)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    featlist.append(featlist_gldm)
    #featlist.append(featlist_ngtdm)
    #featlist.append(featlist_glszm)
    
    
    
    pdf = pd.DataFrame()
    
    
    
    for z in range(0,len(featlist)):
    
        dataset = list()
        
        features = list()
   
        c=0
        
        for q in featlist[z]:
            label1=featureVector[q]
            label2=featureVector2[q]
            label3=featureVector3[q]
            Adder = sitk.AddImageFilter()
            Bidder=Adder.Execute(label1,label2)
            res=Adder.Execute(Bidder,label3)
            #img_view =numpy.empty(30,30,30)
            img_views=sitk.GetArrayViewFromImage(res)
            features.append(copy.copy(img_views))
            c=c+1
    
        
        #normalize the features
        
        print("Ciao Ciaoo skoupidopaido Ypatia")
        
        #dims=[30,30,30]
        dims = label1.GetSize()
        size = dims[2]*dims[1]*dims[0]
        #gdoup = numpy.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
        del label1,label2,label3,Adder,Bidder,res 
    
        #for i in range(0,len(features)):
        #    features[i] = (features[i]-numpy.min(features[i]))/ (numpy.max(features[i])-numpy.min(features[i]))
        mask_view = sitk.GetArrayFromImage(mask)
        bright_view = sitk.GetArrayFromImage(image)
        
        
        for i in range(0,dims[2]):
            for j in range(0,dims[1]):
                for k in range(0,dims[0]):
                    temp = list()
                    temp.append(mask_view[i][j][k]) #pairnoume ti maska gia na skiparoume mideniki klasi sto dataset
                    for q in range(0,len(features)):
                        if(features[q][i][j][k]!='nan'):
                            temp.append(features[q][i][j][k])
                    #temp.append(bright_view[i][j][k])
                    #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                    #temp.append(dist)
                    if mask_view[i][j][k] != 0: #skipparoume mideniki klasi
                        dataset.append(temp) 
        
        ###NORMALIZATION###    
                        
        dataset = np.array(dataset)
        
        for q in range(1, len(dataset[0])):
            min_value = np.min(dataset[:,q])
            max_value = np.max(dataset[:,q])
    
            dataset[:,q] = (dataset[:,q] - min_value) / (max_value - min_value)
       
        ###NORMALIZATION###    
        
        for l in range(1,len(dataset[0])):
            class1 = list()
            class2 = list()
            class3 = list()
            for d in range(0,len(dataset)):
            #for q in range(0,1): 
                 
                 #pltcnt += 1
                 #for d in range(0,len(dataset)-1):
                 if dataset[d][0] == 1:
                     class1.append(dataset[d][l])
                 elif dataset[d][0] == 2:    
                     class2.append(dataset[d][l])
                 elif dataset[d][0] == 3:      
                     class3.append(dataset[d][l])
                 
               
            plt.figure(1)
            plt.hist(class1,bins=100,color='blue')
            
            plt.figure(2)
            plt.hist(class2,bins=100,color='red')
            plt.figure(3)
            plt.hist(class3,bins=100,color='green')
            
            class0 = list()
            class0.append(class1)
            class0.append(class2)
            class0.append(class3)
            plt.figure(4)             
            colors = ['green', 'blue', 'lime']
            labels = ['1','2','3']
            plt.hist(class0,bins=100,color=colors,label=labels,density=True)
            plt.legend(prop ={'size': 10})
             
            
            plt.title(featlist[z][l-1],fontweight ="bold")
            
            #plt.title('brightness 3erwgw''\n\n',fontweight ="bold")
            #plt.title('matplotlib.pyplot.hist() function Example\n\n',fontweight ="bold")
             
            plt.show()
       