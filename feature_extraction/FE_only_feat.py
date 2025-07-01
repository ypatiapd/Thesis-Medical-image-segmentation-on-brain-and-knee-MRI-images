# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:46:33 2022

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:43:34 2022

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
import SimpleITK as sitk
import six
import click
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

import dipy 
import warnings


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



def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
#13,28,39,33,23,22,16,11,4,20 #most similar images

#imgs=['14','17','18','19','21','22','23','25','27','28','29','30','31','32','33','35','37','40','41','42']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','15','16','20','26','34','38','39']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','3','3','3','3','3','3','3']
#imgs=['26','20','13','39',
#imgs=['33','23','22']#,
#imgs=['13']#,'28','22','23']
#n=['3','3','4','3',
#n=['4','4','4']#,
#n=['4']#,'4','4','4']
#imgs=['01','11']
#n=['4','4']
#imgs=['14','17','20','21','29',
#imgs=['06','07','09','14','21','25',
imgs=['04','09','20','38','14','22','17','06','26','07','09','29','25']
#imgs=[]

n=['4','4','3','3','4','4','4','4','3','3','4','4','4']#,'4','4']
#n=[]#,'4','4']

#imgs=['28']
#n=['4']
counter=0

for ite in imgs:  
    
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised_and_norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
    imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised_and_norm_imgs/norm'+ite+'.hdr'
    maskName='D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    paramsFile = 'C:/Users/ypatia/diplomatiki/params-1.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    image_arr = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(maskName)
    mask_arr = sitk.GetArrayFromImage(mask)
    

    #cropped mask gia grigora tests 
    ##uncoment for ROI 
    '''
    a=sitk.RegionOfInterestImageFilter()
    
    #a.SetRegionOfInterest([90,68,60,10,10,10])
    #a.SetRegionOfInterest([60,102,60,30,34,30])
    a.SetRegionOfInterest([66,130,88,22,26,22])
    #a.SetRegionOfInterest([90,68,60,30,34,30])
    #a.SetRegionOfInterest([60,68,90,30,34,30])
    #a.SetRegionOfInterest([88,104,132,10,10,10])
    #a.SetRegionOfInterest([88,104,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([60,72,60,88,104,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    
    #a.SetRegionOfInterest([44,52,44,88,104,88])
    
    mask=a.Execute(mask)
    image=a.Execute(image) 
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)'''
    '''maskArr = sitk.GetArrayFromImage(mask)
    imagArr = sitk.GetArrayFromImage(image)
    
    maskArr[0][0][0] = 3
    maskArr[43][51][20]=3  # to anapodo(kathreptismenes oi diastaseis) apo to bb toy fV 
    
    imagArr[0][0][0] = 350
    imagArr[43][51][20]=350
    
    mask = sitk.GetImageFromArray(maskArr)
    image = sitk.GetImageFromArray(imagArr)'''
    
    
    #setings = {}
    #setings['minimumROIsize'] = 44*44*52
    #setings['minimumROIDimensions'] = 3
    
    
    applyLog = False
    applyWavelet = False
    
    # Setting for the feature calculation.
    # Currently, resampling is disabled.
    # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
    '''settings = {'binWidth': 50,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None}
    
    #
    # If enabled, resample image (resampled image is automatically cropped.
    #
    interpolator = settings.get('interpolator')
    resampledPixelSpacing = settings.get('resampledPixelSpacing')
    if interpolator is not None and resampledPixelSpacing is not None:
      image, mask = imageoperations.resampleImage(image, mask)
    
    bb, correctedMask = imageoperations.checkMask(image, mask,correctMask = "True")#,**setings)
    if correctedMask is not None:
      mask = correctedMask
    #image, mask = imageoperations.cropToTumorMask(image, mask, bb)
    
    '''
    '''gradient_x = np.gradient(image_arr, axis=0)
    gradient_y = np.gradient(image_arr, axis=1)
    gradient_z = np.gradient(image_arr, axis=2)
    
    # Compute the magnitude and direction of the gradient for every voxel
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y) + np.square(gradient_z))
    gradient_direction = np.arctan2(np.sqrt(np.square(gradient_y) + np.square(gradient_z)), gradient_x)
    
    #gradient_direction = np.arctan2(np.sqrt(np.square(gradient_y) + np.square(gradient_z)), gradient_x)
 
    # Define the number of orientation bins
    n_bins = 8
    #n_bins = 26
    
    # Define the range of the orientation histogram
    bin_range = (0, 2 * np.pi)
    
    # Compute the histogram for each voxel
    orientation_histogram = np.zeros((n_bins, gradient_direction.shape[0], gradient_direction.shape[1], gradient_direction.shape[2]))
    for i in range(gradient_direction.shape[0]):
        for j in range(gradient_direction.shape[1]):
            for k in range(gradient_direction.shape[2]):
                bin_number = np.digitize(gradient_direction[i, j, k], np.linspace(*bin_range, n_bins + 1))
                orientation_histogram[bin_number - 1, i, j, k] += gradient_magnitude[i, j, k]
    
    # Compute the mean orientation histogram for each voxel
    mean_orientation_histogram = np.mean(orientation_histogram, axis=0)'''
    
    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=25,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=25,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=25,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=3)
    
    c1 = next(L1)
    level1= c1[0]   
    b1 =next(L1)
    level2= b1[0]
    a1=next(L1)
    kyr1=a1[0]
    lbp_arr1 = sitk.GetArrayFromImage(level1)    
    kyr_arr1=  sitk.GetArrayFromImage(kyr1)    
    c2 = next(L2)
    level21= c2[0]
    b2 =next(L2)
    level22= b2[0] 
    a2=next(L2)
    kyr2=a2[0]
    lbp_arr21 = sitk.GetArrayFromImage(level21)
    kyr_arr2=  sitk.GetArrayFromImage(kyr2)    
    c3 = next(L3)
    level31= c3[0]
    b3 =next(L3)
    level32= b3[0] 
    a3=next(L3)
    kyr3=a3[0]
    lbp_arr31 = sitk.GetArrayFromImage(level31)
    kyr_arr3=  sitk.GetArrayFromImage(kyr3)    

    
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
    
    kyr_arr = np.zeros((len(kyr_arr1), len(kyr_arr1[0]), len(kyr_arr1[0][0])))
    for i in range(0,len(kyr_arr1)):
        for j in range(0,len(kyr_arr1[0])):
            for k in range(0,len(kyr_arr1[0][0])):
                
                if kyr_arr1[i][j][k]!=0:
                    kyr_arr[i][j][k]=kyr_arr1[i][j][k]
                elif kyr_arr2[i][j][k]!=0:
                    kyr_arr[i][j][k]=kyr_arr2[i][j][k]
                elif kyr_arr3[i][j][k]!=0:
                    kyr_arr[i][j][k]=kyr_arr3[i][j][k] 
    
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
    #paratiroume oti i eikona me tis times twn features pou epistrefetai exei max megethos oso i maska pou
    #pername(ROI) , alla se periptwsi pou exei trigyrw pixels klasis 0 , epistrefei mikroteres diastaseis pinaka.
    #otan pername oli ti maska, exei idies diastaseis me ti maska kai tin eikona ,diorthwmenes
    #a1=featureVector['original_firstorder_Mean']
    #a2=featureVector2['original_firstorder_Mean']
    #a3=featureVector3['original_firstorder_Mean']
    
    
    #kala feat glcm : Autocorrelation,jointAverage
    #kala feat glrlm:  LongRunHighGrayLevelEmphasis,HighGrayLevelRunEmphasis,ShortRunHighGrayLevelEmphasis,
    #kala gldm : grayLevelVariance(kaloutsiko),HighGrayLevelEmphasis
    featlist = list()
    featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy']
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Kurtosis','original_firstorder_Maximum','original_firstorder_Mean','original_firstorder_MeanAbsoluteDeviation','original_firstorder_Median','original_firstorder_Minimum','original_firstorder_Minimum','original_firstorder_Range','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_TotalEnergy','original_firstorder_Uniformity','original_firstorder_Variance']
    #featlist_glcm = ['original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_Correlation','original_glcm_MaximumProbability','original_glcm_Id','original_glcm_DifferenceAverage'] 
    featlist_glcm = ['original_glcm_Autocorrelation']#,'original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 

    featlist_glrlm = ['original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis']#,'original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
    
    featlist_gldm = ['original_gldm_HighGrayLevelEmphasis','original_gldm_LowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']#,'original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    #featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    featlist_glszm = ['original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_LargeAreaLowGrayLevelEmphasis','original_glszm_LargeAreaHighGrayLevelEmphasis']#,'original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis']#'original_glszm_LargeAreaLowGrayLevelEmphasis','log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis']
    #featlist_shape= ['original_shape_MeshVolume','original_shape_VoxelVolume']
    
    #,'original_glcm_Correlation'
    
    
    #featlist.append(featlist_shape)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    featlist.append(featlist_gldm)
    featlist.append(featlist_firstorder)
    #featlist.append(featlist_glszm)
    
    
    
    pdf = pd.DataFrame()
    
    
    
    for z in range(0,len(featlist)):
    
        dataset = list()
        
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
        
        if z<len(featlist)-1:
            for i in range(0,dims[2]):
                for j in range(0,dims[1]):
                    for k in range(0,dims[0]):
                        temp = list()
                        temp.append(mask_view[i][j][k]) #pairnoume ti maska gia na skiparoume mideniki klasi sto dataset
                        for q in range(0,len(features)):
                            temp.append(features[q][i][j][k])
                        #temp.append(bright_view[i][j][k])
                        #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                        #temp.append(dist)
                        if mask_view[i][j][k] != 0: #skipparoume mideniki klasi
                            dataset.append(temp) 
                            
                        '''if not numpy.isnan(temp[0]):
                            dataset.append(temp)'''
        else:   
            for i in range(0,dims[2]):
                for j in range(0,dims[1]):
                    for k in range(0,dims[0]):
                        temp = list()
                        temp.append(mask_view[i][j][k])
                        temp.append(bright_view[i][j][k])
                        temp.append(lbp_arr[i][j][k])
                        #temp.append(kyr_arr[i][j][k])
                        #temp.append(mean_orientation_histogram[i][j][k])
                        #temp.append(gradient_magnitude[i][j][k])
                        #temp.append(gradient_direction[i][j][k])
                        temp.append(i)
                        temp.append(j)
                        temp.append(k)
                        spatial_location=np.linalg.norm(np.array([i,j,k]),axis=0)
                        temp.append(spatial_location)
                        for q in range(0,len(features)):
                            temp.append(features[q][i][j][k])
                        
                        #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                        #temp.append(dist)
                        if mask_view[i][j][k] != 0:
                            dataset.append(temp)    
                            
                        '''if not numpy.isnan(temp[0]):
                            dataset.append(temp)'''
        
        #del dataset[dataset[:][0]=='NaN']
        #del dataset[-1]
        #arrdf = numpy.array(dataset)
       
        
        #NAN STUFF ME KEFALAIA GIA NA TA BLEPOUME KALUTERA, mprabo   
        #df2.isnull().values.any()
        #dataset.isnull().sum().sum()
        #dataset.dropna()
        
        dataset = pd.DataFrame(dataset)
        print(len(dataset))
        dataset=dataset.dropna()
        print(len(dataset))
        
        if z<len(featlist)-1: #z<2
            
            df2 = dataset.loc[:,1:len(featlist[z])+1].values #pairnw oles tis stiles ektos apo tin prwti pou einai oi klaseis 
            #df = StandardScaler().fit_transform(df2)
            
            #pca = PCA(n_components=5)
            #principalComponents = pca.fit_transform(df)
            principalDf = pd.DataFrame(data = df2)
                         #, columns = [str(5*z+8), str(5*z+9), str(5*z+10),str(5*z+11), str(5*z+12)])#,str(7*z+10),str(7*z+11),str(7*z+12)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
        
            pdf = pd.concat([pdf,principalDf], axis=1)
            
            del df2,dataset,principalDf
            
        else:
            
            df_im = pd.DataFrame(dataset.loc[:,0:6].values,columns = [str(0), str(1) , str(2),str(3), str(4) , str(5),str(6)]) #stin teleutaia epanalipsi pairnw tis treis prwtes steiles pou einai oi klaseis, to brightness kai lbp 
            
            df2 = dataset.loc[:,7:len(featlist[z])+7].values #pairnw kai tis upoloipes pou einai ta xaraktiristika gia pca
          
            principalDf = pd.DataFrame(data = df2)
            #             , columns = [str(5*z+8), str(5*z+9), str(5*z+10),str(5*z+11), str(5*z+12)])#,str(7*z+10),str(7*z+11),str(7*z+12)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
            
            pdf = pd.concat([pdf,principalDf], axis=1)
          
            pdf = pd.concat([df_im,pdf], axis=1) # prosthetw sto pdf ta pca components kai tis 3 prwtes stiles
            
            
    for i in range(1,len(pdf.columns)):
        pdf[pdf.columns[i]] = (pdf[pdf.columns[i]]-np.min(pdf[pdf.columns[i]]))/ (np.max(pdf[pdf.columns[i]])-np.min(pdf[pdf.columns[i]]))     
    
    with open('file'+ite+'.txt', 'w') as output:
        for row in range(0,len(pdf)):
            s = ",".join(map(str, pdf.iloc[row]))
            output.write(str(s) + '\n')
            
    counter+=1
    
    
    