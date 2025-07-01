# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:44:36 2022

@author: ypatia
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


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#4,6,9,13,20,26,34,38,39,16   label1
#6,7,9,11,13,16,20,26,38,39

imgs=['01','02','03','04','05','06','07','09','10','11','12','13','15','16','20','26','34','38','39']
n=['4','4','4','4','4','4','3','4','4','4','4','4','3','3','3','3','3','3','3']
classes=[1,2,3]
mse_all=list()
for c in classes:
    fo_list=list()
    counter=0
    for ite in imgs:   
        
        imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
        maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
        paramsFile = 'C:/Users/ypatia/diplomatiki/params-1.yaml'
        #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n4_anon_sbj_111.hdr'
        #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/FSL_SEG/OAS1_00'+z+'_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
        #paramsFile = 'C:/Users/ypatia/diplomatiki/params.yaml'
        
        
        if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
          print('Error getting testcase!')
          exit()
        
        image = sitk.ReadImage(imageName)
        mask1 = sitk.ReadImage(maskName)
        
        mask=mask1
        
        maskArr = sitk.GetArrayFromImage(image)
        print(maskArr.mean())
          #comment if ROI
        
        applyLog = False
        applyWavelet = False
        
        # Setting for the feature calculation.
        # Currently, resampling is disabled.
        # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
        settings = {'binWidth': 250,
                    'interpolator': sitk.sitkBSpline,
                    'resampledPixelSpacing': None}
        
        #
        # If enabled, resample image (resampled image is automatically cropped.
        #
        interpolator = settings.get('interpolator')
        resampledPixelSpacing = settings.get('resampledPixelSpacing')
        if interpolator is not None and resampledPixelSpacing is not None:
          image, mask = imageoperations.resampleImage(image, mask, **settings)
        
        bb, correctedMask = imageoperations.checkMask(image, mask,correctMask = "True")#,**setings)
        if correctedMask is not None:
          mask = correctedMask
        #image, mask = imageoperations.cropToTumorMask(image, mask, bb)
        
        # Get the PyRadiomics logger (default log-level = INFO
        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
        
        # Write out all log entries to a file
        handler = logging.FileHandler(filename='testLog.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Initialize feature extractor using the settings file
        '''extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
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
        
        #all_features=numpy.zeros((20,20,20))'''
        
        
        #featureVector = extractor.execute(image, mask,label=1, voxelBased=False)
        #featlist_firstorder = ['original_firstorder_Mean','original_firstorder_StandardDeviation','original_firstorder_TotalEnergy','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_Minimum','original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Maximum','original_firstorder_Median','original_firstorder_InterquartileRange','original_firstorder_Range','original_firstorder_MeanAbsoluteDeviation','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_Kurtosis','original_firstorder_Variance','original_firstorder_Uniformity'] 
        #for q in featlist_firstorder:
        #    feature=featureVector[q]
        #    print(feature)
        #['Mean','StandardDeviation','TotalEnergy','Energy','Entropy','Minimum','10Percentile','90Percentile','Maximum','Median','InterquartileRange','Range','MeanAbsoluteDeviation','RobustMeanAbsoluteDeviation','RootMeanSquared','Skewness','Kurtosis','Variance','Uniformity']
        
        
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask,label=c)
        
        #a=firstOrderFeatures.getMedianFeatureValue()
        #print(a)
        #firstOrderFeatures.enableFeatureByName('Mean',True)
        #firstOrderFeatures.enableAllFeatures()
        
        print('Will calculate the following first order features: ')
        for f in firstOrderFeatures.enabledFeatures.keys():
          print('  ', f)
          print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)
        
        print('Calculating first order features...')
        results = firstOrderFeatures.execute()
        print('done')
        featlist_firstorder = ['Mean','TotalEnergy','Energy','Entropy','Minimum','10Percentile','90Percentile','Maximum','Median','InterquartileRange','Range','MeanAbsoluteDeviation','RobustMeanAbsoluteDeviation','RootMeanSquared','Skewness','Kurtosis','Variance','Uniformity'] 
        temp=list()
        for q in featlist_firstorder:
            feature=float(results[q])
            print(feature)
            temp.append(feature)
        fo_list.append(temp)
        counter+=1
        
    all_normed = list()
    for i in range(0,len(fo_list[0])):
        goos = list()
    
        for j in range(0,len(fo_list)):
            
            goos.append(fo_list[j][i])
        
        normed = list()
        for k in range(0,len(goos)):
            normed.append((goos[k]-min(goos))/(max(goos)-min(goos)))
        all_normed.append(normed)    
    
    mse=list()
    for j in range(1,len(all_normed[0])):
        sum1=0
        for i in range(0,len(all_normed)):
            sum1+=pow(all_normed[i][0]-all_normed[i][j],2)
        mse.append(sum1/len(all_normed[0]))
    mse_all.append(mse)
mse_total=list()
for i in range(0,len(mse_all[1])):
    mse_total.append(mse_all[1][i]+mse_all[2][i])
    
