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
imgs=['05']
n=['4']

counter=0

for ite in imgs:   
    maskName='C:/Users/jaime/YanAlgorithm/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'    
    imageName='C:/Users/jaime/Desktop/normibet/normalized05.hdr'
    paramsFile = 'C:/Users/jaime/YanAlgorithm/params-1.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    '''
    a=sitk.RegionOfInterestImageFilter()
    
    a.SetRegionOfInterest([45,60,45,10,10,10])
    #a.SetRegionOfInterest([60,102,60,30,34,30])
    #a.SetRegionOfInterest([66,130,88,22,26,22])
    #a.SetRegionOfInterest([90,68,60,30,34,30])
    #a.SetRegionOfInterest([60,68,90,30,34,30])
    #a.SetRegionOfInterest([88,104,132,10,10,10])
    #a.SetRegionOfInterest([88,104,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([60,72,60,88,104,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    
    #a.SetRegionOfInterest([44,52,44,88,104,88])
    
    mask=a.Execute(mask)
    image=a.Execute(image)      
    '''
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask)
    
    bbox = np.array(lsif.GetBoundingBox(1))
    bbox2 = np.array(lsif.GetBoundingBox(2))
    bbox3 = np.array(lsif.GetBoundingBox(3))
    
    
    bboxs = np.vstack((bbox,bbox2))
    bboxs = np.vstack((bboxs,bbox3))
    #print(bboxs)
    
    minx = 100000 # x OXI oti bazw sthn prwth 8esh tou maskArr
    miny = 100000 # y oti bazw sthn deuterh 8esh tou maskArr
    minz = 100000 # z OXI oti bazw sthn trith 8esh tou maskArr
    maxx = 0
    maxy = 0
    maxz = 0
    pminx,pminy,pminz = list(),list(),list()
    pmaxx,pmaxy,pmaxz = list(),list(),list()
    

    
    if np.all(bbox == bbox2) and np.all(bbox2 == bbox3):
        print("ALL GOOD")
    else:
        for i in range(0,2):
            
            if bboxs[i][0]<=minx:
                minx=bboxs[i][0]
                pminx.append(i)
            if bboxs[i][1]<=miny:
                miny=bboxs[i][1]
                pminy.append(i)
            if bboxs[i][2]<=minz:
                minz=bboxs[i][2]
                pminz.append(i)
            if bboxs[i][0]+bboxs[i][3]-1 >= maxx:
                maxx = bboxs[i][0]+bboxs[i][3]-1
                pmaxx.append(i)              
            if bboxs[i][1]+bboxs[i][4]-1 >= maxy:
                maxy = bboxs[i][1]+bboxs[i][4]-1
                pmaxy.append(i)          
            if bboxs[i][2]+bboxs[i][5]-1 >= maxz:
                maxz = bboxs[i][2]+bboxs[i][5]-1
                pmaxz.append(i)              
    

        maskArr = sitk.GetArrayFromImage(mask)         
        maskArr[minz][miny][minx] = 3
        maskArr[maxz][maxy][maxx] = 3
        maskArr[minz][maxy][minx] = 2
        maskArr[maxz][miny][maxx] = 2
        maskArr[maxz][miny][minx] = 1
        maskArr[minz][maxy][maxx] = 1
        mask = sitk.GetImageFromArray(maskArr)
        
        image_arr = sitk.GetArrayFromImage(image)
        
        image.SetOrigin([0,0,0])
    
    
    
    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=3)
    
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
   
    
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Kurtosis','original_firstorder_Maximum','original_firstorder_Mean','original_firstorder_MeanAbsoluteDeviation','original_firstorder_Median','original_firstorder_Minimum','original_firstorder_Minimum','original_firstorder_Range','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_TotalEnergy','original_firstorder_Uniformity','original_firstorder_Variance']
    #featlist_glcm = ['original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_Correlation','original_glcm_MaximumProbability','original_glcm_Id','original_glcm_DifferenceAverage'] 
    #,'original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 

    #,'original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    featlist = list()
    featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy']  #'original_glcm_JointAverage'
    featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
    featlist_gldm = ['original_gldm_SmallDependenceEmphasis','original_gldm_LargeDependenceEmphasis','original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized','original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis','original_glszm_LargeAreaLowGrayLevelEmphasis','original_glszm_LargeAreaHighGrayLevelEmphasis']
    #featlist_shape= ['original_shape_MeshVolume','original_shape_VoxelVolume']
    
    '''
    featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_JointAverage']
    featlist_gldm = ['original_gldm_GrayLevelVariance','original_gldm_HighGrayLevelEmphasis']
    '''
    
    
    '''featlist.extend(featlist_glcm)
    featlist.extend(featlist_glrlm)
    featlist.extend(featlist_gldm)
    featlist.extend(featlist_firstorder)'''
       
    
    #featlist.append(featlist_shape)
    '''featlist.append(featlist_glcm)
    featlist.append(featlist_gldm)
    featlist.append(featlist_firstorder)'''
    featlist.append(featlist_firstorder)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    featlist.append(featlist_gldm)
    featlist.append(featlist_ngtdm)
    featlist.append(featlist_glszm)

    #featlist.append(featlist_glszm)
    
    
    
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
        
    selector = SelectKBest(score_func=mutual_info_classif, k=30)
    X_new = selector.fit_transform(X, y)
    selected_feature_indices = selector.get_support(indices=True)
    
    with open('Most_important_feat.txt', 'w') as f:
       for item in selected_feature_indices:
           f.write(str(item) + '\n')
   
    