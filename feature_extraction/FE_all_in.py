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
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations


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
imgs=['13','39','28','33','23','11']
n=['4','3','4','4','4','4']

counter=0

for ite in imgs:   
    #imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
    imageName='C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/denoised_and_norm_imgs/norm'+ite+'.hdr'
    
    #maskName='C:/Users/ypatia/diplomatiki/registered_masks/registered_'+ite+'.hdr'
    maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    paramsFile = 'C:/Users/ypatia/diplomatiki/params-1.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    #cropped mask gia grigora tests 
    ##uncoment for ROI 
    
    #a=sitk.RegionOfInterestImageFilter()
    
    #a.SetRegionOfInterest([90,68,60,10,10,10])
    #a.SetRegionOfInterest([60,102,60,30,34,30])
    #a.SetRegionOfInterest([66,130,88,22,26,22])
    #a.SetRegionOfInterest([90,68,60,30,34,30])
    #a.SetRegionOfInterest([60,68,90,30,34,30])
    #a.SetRegionOfInterest([88,104,132,10,10,10])
    #a.SetRegionOfInterest([88,104,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    #a.SetRegionOfInterest([60,72,60,88,104,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
    
    #a.SetRegionOfInterest([44,52,44,88,104,88])
    
    #mask=a.Execute(mask)
    #image=a.Execute(image) 
    
    
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
    settings = {'binWidth': 50,
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
    image, mask = imageoperations.cropToTumorMask(image, mask, bb)
    
    
    L=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2)
    
    c = next(L)
    level1= c[0]  #first level
    b = next(L)
    level2 = b[0]  #second level
    d=next(L) #kyrtosis
    kyrtosis= d[0]
    lbp_arr1 = sitk.GetArrayFromImage(level1)
    lbp_arr2 = sitk.GetArrayFromImage(level2)
    lbp_kyr=   sitk.GetArrayFromImage(kyrtosis)
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
    
    glcmFeatures = glcm.RadiomicsGLCM(image, mask)
    glcmFeatures.enableAllFeatures()
    
    print('Calculating GLCM features...')
    results = glcmFeatures.execute()
    print('done')
    
    print('Calculated GLCM features: ')
    for (key, val) in six.iteritems(results):
      print('  ', key, ':', val)
    print('eftasa')
    featureVector = extractor.execute(image, mask,label=1, voxelBased=True)
    featureVector2 = extractor.execute(image, mask,label=2, voxelBased=True)
    featureVector3 = extractor.execute(image, mask,label=3, voxelBased=True)
    #paratiroume oti i eikona me tis times twn features pou epistrefetai exei max megethos oso i maska pou
    #pername(ROI) , alla se periptwsi pou exei trigyrw pixels klasis 0 , epistrefei mikroteres diastaseis pinaka.
    #otan pername oli ti maska, exei idies diastaseis me ti maska kai tin eikona ,diorthwmenes
    #a1=featureVector['original_firstorder_Mean']
    #a2=featureVector2['original_firstorder_Mean']
    #a3=featureVector3['original_firstorder_Mean']
    
    featlist = list()
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Kurtosis','original_firstorder_Maximum','original_firstorder_Mean','original_firstorder_MeanAbsoluteDeviation','original_firstorder_Median','original_firstorder_Minimum','original_firstorder_Minimum','original_firstorder_Range','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_RootMeanSquared','original_firstorder_Skewness','original_firstorder_TotalEnergy','original_firstorder_Uniformity','original_firstorder_Variance']
    #featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy'] 
    #featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_Correlation','original_glcm_MaximumProbability','original_glcm_SumSquares','original_glcm_Id','original_glcm_DifferenceAverage'] 
    featlist_glcm = ['original_glcm_JointAverage','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_Correlation','original_glcm_Id','original_glcm_DifferenceAverage'] 

    #featlist_glrlm = ['original_glrlm_ShortRunEmphasis','original_glrlm_LongRunEmphasis','original_glrlm_GrayLevelNonUniformity','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
    featlist_glrlm= ['original_glrlm_ShortRunEmphasis','original_glrlm_LongRunEmphasis','original_glrlm_GrayLevelNonUniformity','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy']
    featlist_gldm = ['original_gldm_SmallDependenceEmphasis','original_gldm_LargeDependenceEmphasis','original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    #featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    #featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized']#,'original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis']#'original_glszm_LargeAreaLowGrayLevelEmphasis','log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis']
    #featlist_shape= ['original_shape_MeshVolume','original_shape_VoxelVolume']
    
    #,'original_glcm_Correlation'
    
    #featlist.append(featlist_firstorder)
    #featlist.append(featlist_shape)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    #featlist.append(featlist_gldm)
    #featlist.append(featlist_ngtdm)
    
    
    
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
                        temp.append(lbp_arr1[i][j][k])
                        temp.append(lbp_arr2[i][j][k])
                        temp.append(lbp_kyr[i][j][k])
                        temp.append(i)
                        temp.append(j)
                        temp.append(k)
                        #temp.append(y.GetPixel(i,j,k))
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
            df = StandardScaler().fit_transform(df2)
            
            pca = PCA(n_components=4)
            principalComponents = pca.fit_transform(df)
            principalDf = pd.DataFrame(data = principalComponents
                         , columns = [str(4*z+8), str(4*z+9), str(4*z+10),str(4*z+11)])#,str(7*z+10),str(7*z+11),str(7*z+12)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
        
            pdf = pd.concat([pdf,principalDf], axis=1)
            
            del df,df2,dataset
            del principalComponents
            
        else:
            
            df_im = pd.DataFrame(dataset.loc[:,0:7].values,columns = [str(0), str(1) , str(2),str(3), str(4) , str(5),str(6),str(7)]) #stin teleutaia epanalipsi pairnw tis treis prwtes steiles pou einai oi klaseis, to brightness kai lbp 
            
            df2 = dataset.loc[:,8:len(featlist[z])+8].values #pairnw kai tis upoloipes pou einai ta xaraktiristika gia pca
            df = StandardScaler().fit_transform(df2)
            
            pca = PCA(n_components=4)
            principalComponents = pca.fit_transform(df)
            principalDf = pd.DataFrame(data = principalComponents
                         , columns = [str(4*z+8), str(4*z+9), str(4*z+10),str(4*z+11)])#,str(7*z+10),str(7*z+11),str(7*z+12)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
            
            pdf = pd.concat([pdf,principalDf], axis=1)
            pdf = pd.concat([df_im,pdf], axis=1) # prosthetw sto pdf ta pca components kai tis 3 prwtes stiles
            #del df,df1,df2,df3
            #del principalComponents'''
            
    #pdf = pdf.loc[(pdf["0"]!=0)]  #apokleioume ti mideniki klasi ap to dataset
            
    for i in range(1,len(pdf.columns)):
        pdf[pdf.columns[i]] = (pdf[pdf.columns[i]]-numpy.min(pdf[pdf.columns[i]]))/ (numpy.max(pdf[pdf.columns[i]])-numpy.min(pdf[pdf.columns[i]]))     
    
    with open('file'+ite+'.txt', 'w') as output:
        for row in range(0,len(pdf)):
            s = ",".join(map(str, pdf.iloc[row]))
            output.write(str(s) + '\n')
            
    counter+=1
    
    
    