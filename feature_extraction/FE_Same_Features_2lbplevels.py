# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:29:51 2022

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:35:31 2022

@author: ypatia
"""

"""
Created on Thu Apr 21 12:29:32 2022

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

imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
paramsFile = 'C:/Users/ypatia/diplomatiki/params.yaml'

if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()

image = sitk.ReadImage(imageName)
mask1 = sitk.ReadImage(maskName)


#cropped mask gia grigora tests 
a=sitk.RegionOfInterestImageFilter()

#a.Image=image

#a.SetRegionOfInterest([66,130,88,22,26,22])
a.SetRegionOfInterest([60,102,60,30,34,30])
#a.SetRegionOfInterest([90,68,60,30,34,30])
#a.SetRegionOfInterest([88,104,132,10,10,10])
#a.SetRegionOfInterest([88,52,88,44,52,44])   #anapoda oi akrianes diastaseis sti maska !!!!!!!!!!!!!!!!!!!!!!!
#a.SetRegionOfInterest([88,104,88,44,52,44])




#a.SetSize([5,5,5]) #size of ROI
#a.SetIndex([110,110,80])
#a.SetRegionOfInterest([110,110,80])

mask=a.Execute(mask1)
image=a.Execute(image) 

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


L=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=50,lbp3DIcosphereRadius=2,lbp3DLevels=2)

c = next(L)
#b = next(L)
y = c[0]
b = next(L)
y2 = b[0]

lbp_arr = sitk.GetArrayFromImage(y)
lbp_arr2 = sitk.GetArrayFromImage(y2)


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
#paratiroume oti i eikona me tis times twn features pou epistrefetai exei max megethos oso i maska pou
#pername(ROI) , alla se periptwsi pou exei trigyrw pixels klasis 0 , epistrefei mikroteres diastaseis pinaka.
#otan pername oli ti maska, exei idies diastaseis me ti maska kai tin eikona ,diorthwmenes
#a1=featureVector['original_firstorder_Mean']
#a2=featureVector2['original_firstorder_Mean']
#a3=featureVector3['original_firstorder_Mean']

featlist = list()
featlist_glcm = ['log-sigma-1-0-mm-3D_glcm_Autocorrelation','log-sigma-1-0-mm-3D_glcm_DifferenceEntropy','log-sigma-1-0-mm-3D_glcm_ClusterProminence','log-sigma-1-0-mm-3D_glcm_JointAverage','log-sigma-1-0-mm-3D_glcm_ClusterTendency','log-sigma-1-0-mm-3D_glcm_ClusterShade','log-sigma-1-0-mm-3D_glcm_Contrast','log-sigma-1-0-mm-3D_glcm_DifferenceAverage','log-sigma-1-0-mm-3D_glcm_DifferenceVariance','log-sigma-1-0-mm-3D_glcm_JointEnergy','log-sigma-1-0-mm-3D_glcm_JointEntropy','log-sigma-1-0-mm-3D_glcm_SumSquares'] 
featlist_glrlm = ['log-sigma-1-0-mm-3D_glrlm_ShortRunEmphasis','log-sigma-1-0-mm-3D_glrlm_LongRunEmphasis','log-sigma-1-0-mm-3D_glrlm_GrayLevelNonUniformity','log-sigma-1-0-mm-3D_glrlm_GrayLevelNonUniformityNormalized','log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformity','log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformityNormalized','log-sigma-1-0-mm-3D_glrlm_RunPercentage','log-sigma-1-0-mm-3D_glrlm_GrayLevelVariance','log-sigma-1-0-mm-3D_glrlm_RunVariance','log-sigma-1-0-mm-3D_glrlm_RunEntropy','log-sigma-1-0-mm-3D_glrlm_LowGrayLevelRunEmphasis','log-sigma-1-0-mm-3D_glrlm_HighGrayLevelRunEmphasis','log-sigma-1-0-mm-3D_glrlm_ShortRunLowGrayLevelEmphasis','log-sigma-1-0-mm-3D_glrlm_ShortRunHighGrayLevelEmphasis']#,'log-sigma-1-0-mm-3D_glrlm_LongRunLowGrayLevelEmphasis','log-sigma-1-0-mm-3D_glrlm_LongRunHighGrayLevelEmphasis']
featlist_gldm = ['log-sigma-1-0-mm-3D_gldm_SmallDependenceEmphasis','log-sigma-1-0-mm-3D_gldm_LargeDependenceEmphasis','log-sigma-1-0-mm-3D_gldm_GrayLevelNonUniformity','log-sigma-1-0-mm-3D_gldm_DependenceNonUniformity','log-sigma-1-0-mm-3D_gldm_DependenceNonUniformityNormalized','log-sigma-1-0-mm-3D_gldm_GrayLevelVariance','log-sigma-1-0-mm-3D_gldm_DependenceVariance','log-sigma-1-0-mm-3D_gldm_DependenceEntropy','log-sigma-1-0-mm-3D_gldm_LowGrayLevelEmphasis','log-sigma-1-0-mm-3D_gldm_HighGrayLevelEmphasis','log-sigma-1-0-mm-3D_gldm_SmallDependenceLowGrayLevelEmphasis','log-sigma-1-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis','log-sigma-1-0-mm-3D_gldm_LargeDependenceLowGrayLevelEmphasis','log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis']
featlist_ngtdm = ['log-sigma-1-0-mm-3D_ngtdm_Coarseness','log-sigma-1-0-mm-3D_ngtdm_Contrast','log-sigma-1-0-mm-3D_ngtdm_Busyness','log-sigma-1-0-mm-3D_ngtdm_Complexity']#,'log-sigma-1-0-mm-3D_ngtdm_Strength']
featlist_glszm = ['log-sigma-1-0-mm-3D_glszm_SmallAreaEmphasis','log-sigma-1-0-mm-3D_glszm_LargeAreaEmphasis','log-sigma-1-0-mm-3D_glszm_GrayLevelNonUniformity','log-sigma-1-0-mm-3D_glszm_GrayLevelNonUniformityNormalized','log-sigma-1-0-mm-3D_glszm_SizeZoneNonUniformity','log-sigma-1-0-mm-3D_glszm_SizeZoneNonUniformityNormalized','log-sigma-1-0-mm-3D_glszm_ZonePercentage','log-sigma-1-0-mm-3D_glszm_GrayLevelVariance','log-sigma-1-0-mm-3D_glszm_ZoneVariance','log-sigma-1-0-mm-3D_glszm_ZoneEntropy','log-sigma-1-0-mm-3D_glszm_LowGrayLevelZoneEmphasis','log-sigma-1-0-mm-3D_glszm_HighGrayLevelZoneEmphasis','log-sigma-1-0-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis','log-sigma-1-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis']#'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis','log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis']


#,'original_glcm_Correlation'


featlist.append(featlist_glcm)
featlist.append(featlist_glrlm)
#featlist.append(featlist_gldm)
#featlist.append(featlist_ngtdm)
#featlist.append(featlist_glszm)


pdf = pd.DataFrame()



for z in range(0,2):

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
        
    if z<1:
        for i in range(0,dims[2]):
            for j in range(0,dims[1]):
                for k in range(0,dims[0]):
                    temp = list()
                    #temp.append(mask_view[i][j][k])
                    for q in range(0,len(features)):
                        temp.append(features[q][i][j][k])
                    #temp.append(bright_view[i][j][k])
                    #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                    #temp.append(dist)
                    if not numpy.isnan(temp[0]):
                        dataset.append(temp)
                    
                        
    else:
        mask_view = sitk.GetArrayFromImage(mask)
        bright_view = sitk.GetArrayFromImage(image)
        #bright_view = (bright_view-numpy.min(bright_view))/ (numpy.max(bright_view)-numpy.min(bright_view))
        for i in range(0,dims[2]):
            for j in range(0,dims[1]):
                for k in range(0,dims[0]):
                    temp = list()
                    temp.append(mask_view[i][j][k])
                    temp.append(bright_view[i][j][k])
                    temp.append(lbp_arr[i][j][k])
                    temp.append(lbp_arr2[i][j][k])
                    #temp.append(y.GetPixel(i,j,k))
                    for q in range(0,len(features)):
                        temp.append(features[q][i][j][k])
                    
                    #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                    #temp.append(dist)
                    if not numpy.isnan(temp[0]):
                        dataset.append(temp)
    
    #del dataset[dataset[:][0]=='NaN']
    #del dataset[-1]
    #arrdf = numpy.array(dataset)
   
    
    df = pd.DataFrame()
    
    
    if z<1: #z<2
        for j in range(1,len(dataset)-1):
            df1 = pd.DataFrame(dataset[j]).T
            df = df.append(copy.copy(df1), ignore_index=True)
        
        df = df.loc[:,0:len(featlist[z])].values
        df = StandardScaler().fit_transform(df)
        
        #numpy.nan_to_num(df, copy=False, nan=0.0, posinf=None, neginf=None)
        
        #df = df.replace(numpy.nan, 0)
        #dfnew=df[0:100000]
        
        pca = PCA(n_components=7)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [str(7*z+4), str(7*z+5), str(7*z+6), str(7*z+7),str(7*z+8),str(7*z+9),str(7*z+10)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
    
        pdf = pd.concat([pdf,principalDf], axis=1)
        
        del df

        
    else:
        for j in range(1,len(dataset)-1):
            df1 = pd.DataFrame(dataset[j]).T
            df = df.append(copy.copy(df1), ignore_index=True)
        
        df_im = pd.DataFrame(df.loc[:,0:3].values,columns = [str(0), str(1) , str(2), str(3)])
        df = df.loc[:,4:len(featlist[z])].values
        df = StandardScaler().fit_transform(df)
        
        #numpy.nan_to_num(df, copy=False, nan=0.0, posinf=None, neginf=None)
        
        #dfnew=df[0:100000]
        
        pca = PCA(n_components=7)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [str(7*z+4), str(7*z+5), str(7*z+6), str(7*z+7),str(7*z+8),str(7*z+9),str(7*z+10)])         
        
        pdf = pd.concat([pdf,principalDf], axis=1)
        #df_im[:,1] = StandardScaler().fit_transform(df_im[:,1])

        pdf = pd.concat([df_im,pdf], axis=1)
        del df
        
pdf = pdf.loc[(pdf["0"]!=0)]  #apokleioume ti mideniki klasi ap to dataset
        
for i in range(1,len(pdf.columns)):
    pdf[pdf.columns[i]] = (pdf[pdf.columns[i]]-numpy.min(pdf[pdf.columns[i]]))/ (numpy.max(pdf[pdf.columns[i]])-numpy.min(pdf[pdf.columns[i]]))     
            
with open("file.txt", 'w') as output:
    for row in range(0,len(pdf)):
        s = ",".join(map(str, pdf.iloc[row]))
        output.write(str(s) + '\n')


'''auto1=featureVector['original_glcm_Autocorrelation']
auto2=featureVector2['original_glcm_Autocorrelation']
auto3=featureVector3['original_glcm_Autocorrelation']

A = sitk.AddImageFilter()
B = A.Execute(auto1,auto2)
auto = A.Execute(B,auto3)


auto1=featureVector['original_glcm_JointAverage']
auto2=featureVector2['original_glcm_JointAverage']
auto3=featureVector3['original_glcm_JointAverage']



img_view = sitk.GetArrayViewFromImage(auto)


#joint=featureVector['original_glcm_JointAverage']
a=featureVector['log-sigma-1-0-mm-3D_firstorder_StandardDeviation']
a=featureVector['log-sigma-3-0-mm-3D_firstorder_Mean']
a=featureVector['log-sigma-3-0-mm-3D_firstorder_StandardDeviation']
a=featureVector['log-sigma-1-0-mm-3D-LLH_firstorder_Mean']
a=featureVector['log-sigma-1-0-mm-3D-LLH_firstorder_StandardDeviation']'''

#des an ontws kroparei teis seires pou exoun mono midenika 

'''
jointdims = joint.GetSize()

for i in range(0,20):
    for j in range(0,20):
        for z in range(0,20):
            if a1.GetPixel(i,j,z) !=0:
                all_features[i,j,z]=a1.GetPixel(i,j,z)
            elif  a2.GetPixel(i,j,z) !=0:           
                all_features[i,j,z]=a2.GetPixel(i,j,z)
            elif  a3.GetPixel(i,j,z) !=0:           
                all_features[i,j,z]=a3.GetPixel(i,j,z)
            #print(a.GetPixel(i,j,z))

for i in range(0,20):
    for j in range(0,20):
        for z in range(0,20):
            print(all_features[i,j,z])
            
for i in range(0,137):
    for j in range(0,171):
        for z in range(0,145):
            print(all_features[i,j,z])
            
for featureName, featureValue in six.iteritems(featureVector):
  if isinstance(featureValue, sitk.Image):
    sitk.WriteImage(featureValue, '%s_%s.nrrd' % (image, featureName))
    print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, image, featureName))
  else:
    print('%s: %s' % (featureName, featureValue))'''