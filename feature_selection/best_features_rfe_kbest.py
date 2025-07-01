# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:02:14 2023

@author: ypatia
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:31:00 2023

@author: ypatia
"""
import csv
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:42:19 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:23:00 2023

@author: jaime
"""


from sklearn.ensemble import RandomForestClassifier

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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier

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
imgs=['01']
n=['4']

counter=0

for ite in imgs:   
    imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised_and_norm_imgs/norm'+ite+'.hdr'
    maskName='D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'

    #maskName='C:/Users/ypatia/diplomatiki/registered_masks/registered_'+ite+'.hdr'
    #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    paramsFile = 'C:/Users/ypatia/diplomatiki/params-1.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    '''a=sitk.RegionOfInterestImageFilter()
    
    #a.SetRegionOfInterest([90,68,60,10,10,10])
    #a.SetRegionOfInterest([60,102,60,30,34,30])
    a.SetRegionOfInterest([66,130,88,22,26,22])
    #a.SetRegionOfInterest([90,68,60,30,34,30])
    #a.SetRegionOfInterest([30,30,30,30,34,30])
    #a.SetRegionOfInterest([88,104,132,10,10,10])
    #a.SetRegionOfInterest([88,104,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([60,72,60,88,104,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([44,52,44,88,104,88])
    #a.SetRegionOfInterest([54,38,54,100,100,100])
    mask=a.Execute(mask)
    image=a.Execute(image) 
    '''
    
    
    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=3)
    
    c1 = next(L1)
    level1= c1[0]      
    lbp_arr1 = sitk.GetArrayFromImage(level1)    
    c2 = next(L2)
    level21= c2[0]
    lbp_arr21 = sitk.GetArrayFromImage(level21)
    c3 = next(L3)
    level31= c3[0]
    lbp_arr31 = sitk.GetArrayFromImage(level31)
    
    
    lbp_arr = np.zeros((len(lbp_arr1), len(lbp_arr1[0]), len(lbp_arr1[0][0])))
    for i in range(0,len(lbp_arr1)):
        for j in range(0,len(lbp_arr1[0])):
            for k in range(0,len(lbp_arr1[0][0])):
                if lbp_arr1[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr1[i][j][k]
                elif lbp_arr21[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr21[i][j][k]
                elif lbp_arr31[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr31[i][j][k]  
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
    
    
    #glcm: ['Autocorrelation','DifferenceEntropy','ClusterProminence','JointAverage','ClusterTendency','ClusterShade','Contrast','Correlation','DifferenceAverage','DifferenceVariance','JointEnergy','JointEntropy','SumSquares','Id','MaximumProbability','SumSquares','InverseVariance','SumEntropy','Imc1','Imc2','Idm','MCC','Idmn','Idn']

    #ALL FEATURES
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy','original_firstorder_Variance','original_firstorder_Uniformity','original_firstorder_Kurtosis','original_firstorder_Skewness','original_firstorder_MeanAbsoluteDeviation','original_firstorder_Range','original_firstorder_RobustMeanAbsoluteDeviation' ]
    #featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation']#,'original_glcm_Id','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_SumEntropy','original_glcm_Imc1','original_glcm_Imc2','original_glcm_Idm','original_glcm_MCC','original_glcm_Idmn','original_glcm_Idn']  #'original_glcm_JointAverage'
    #featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis']
    #featlist_gldm = ['original_gldm_SmallDependenceEmphasis','original_gldm_LargeDependenceEmphasis','original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    #featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    #featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized','original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis','original_glszm_LargeAreaLowGrayLevelEmphasis','original_glszm_LargeAreaHighGrayLevelEmphasis']
    
    featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_Idn']#,'original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 
    featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis']#,'original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']    
    featlist_gldm = ['original_gldm_LargeDependenceHighGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_LowGrayLevelEmphasis']#,'original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']

    
    
    #TRIED COMBO
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    #featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_JointAverage']#,'original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 
    #featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis']#,'original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']    
    #featlist_gldm = ['original_gldm_GrayLevelVariance','original_gldm_HighGrayLevelEmphasis']#,'original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']

    
    #featlist.append(featlist_shape)
    featlist.append(featlist_firstorder)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    featlist.append(featlist_gldm)
    #featlist.append(featlist_ngtdm)
    #featlist.append(featlist_glszm)
    
    all_features=list()
    all_features.append('Brightness')
    all_features.append('LBP')
    all_features.append('X')
    all_features.append('Y')
    all_features.append('Z')
    all_features.append('dist')
    for  z in range(0,len(featlist)):
        all_features.extend(featlist[z])
    
    all_dataset=list()
    y=list()
    for z in range(0,len(featlist)):
      
        dataset=list()
        features = list()
        #img_views=list()
        #label1=list()
        #label2=list()
        #label3=list()
        #Bidder=list()
        #res=list()
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
                    if z==0 and mask_view[i][j][k]!=0:
                        y.append(mask_view[i][j][k])
                    if z==0:
                        temp.append(bright_view[i][j][k])
                        temp.append(lbp_arr[i][j][k])
                        temp.append(i)
                        temp.append(j)
                        temp.append(k)   
                        spatial_location=np.linalg.norm(np.array([i,j,k]),axis=0)
                        temp.append(spatial_location)
                    #temp.append(mask_view[i][j][k]) #pairnoume ti maska gia na skiparoume mideniki klasi sto dataset
                    for q in range(0,len(features)):
                        temp.append(features[q][i][j][k])
                    #temp.append(bright_view[i][j][k])
                    #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                    #temp.append(dist)
                    if mask_view[i][j][k] != 0: #skipparoume mideniki klasi
                        dataset.append(temp) 
        
        if(z==0):
            all_dataset=dataset
        else:
            all_dataset = np.column_stack((all_dataset, dataset)).tolist()
             
    excluded = list()
    for p in range(0,len(all_dataset)):
        if np.isnan(all_dataset[p]).any():
            excluded.append(p)
    
    ite = 0;
    
    for p in excluded:
        del y[p-ite]
        del all_dataset[p-ite]
        ite+=1
    
    X = pd.DataFrame(all_dataset)
    Y = pd.DataFrame(y)
    
    for i in range(0,len(X.columns)):
        X[X.columns[i]] = (X[X.columns[i]]-np.min(X[X.columns[i]]))/ (np.max(X[X.columns[i]])-np.min(X[X.columns[i]]))     
       

   
    X = X.values
    Y = Y.values 
    Y=Y.ravel()
    
    values=[5,10,15,16,17,18,19,20,24]
    #kbest
    for i in values:
        selector = SelectKBest(score_func=mutual_info_classif, k=i)
        X_new = selector.fit_transform(X, Y)
        selected_feature_indices = selector.get_support(indices=True)  
        selected_features_kbest = np.array(all_features)[selected_feature_indices]
        
        with open('final_best_kbest'+format(i)+'.txt', 'w') as f:
            # Loop through each string in the array
            for s in selected_features_kbest:
                # Write the string to the file followed by a newline character
                f.write(s + '\n')
            
   
            
    
    '''        
    #RFE
    # Create an instance of SVC
    estimator = SVC(kernel="linear")
    
    # Create an instance of RFE
    selector = RFE(estimator, n_features_to_select=25)
    
    # Fit the RFE to the data
    selector = selector.fit(X, Y)
    
    # Print the selected features
    print(selector.support_)
    
    print(selector.ranking_)
    
    selected_features_rfe = np.array(all_features)[selector.support_]
   
    # Open a file for writing (use "w" mode for writing)
    with open('best_rfe.txt', 'w') as f:
        # Loop through each string in the array
        for s in selected_features_rfe:
            # Write the string to the file followed by a newline character
            f.write(s + '\n')
    '''
    #forest RFE#
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    
    # Use RFE to select the top 3 features
    rfe = RFE(clf, n_features_to_select=24)
    X_new = rfe.fit_transform(X, Y)
    
    # Fit the classifier with the data
    clf.fit(X_new, Y)
    
    #print(rfe.support_)
    # Obtain the feature mask
    mask = rfe.support_
    
    # Obtain the feature importance values
    importance = clf.feature_importances_
    
    importance = [format(x, 'f') for x in importance]
    
    selected_features = np.array(all_features)[mask]
    # Print the results
    #print("Selected features: ", iris.feature_names[mask])
    #print("Feature importances: ", importance)
    combined = np.array([selected_features, importance]).T
    best_feat_rfe_tree = combined[combined[:,1].argsort()[::-1]]
    
    with open('final_best_rfe_tree.csv', 'w', newline='') as f:
         # Create a CSV writer object
         writer = csv.writer(f)
         # Loop through each row in the 2D array
         for row in best_feat_rfe_tree:
             # Write the row to the CSV file
             writer.writerow(row)
