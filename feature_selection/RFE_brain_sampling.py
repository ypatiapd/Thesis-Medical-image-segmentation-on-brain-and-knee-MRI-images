import os
import SimpleITK as sitk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import time 
import numpy as np
import six
import dipy 
import warnings
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom
import random
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import logging
import math
import pandas as pd
import radiomics
from radiomics import featureextractor, getFeatureClasses
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape
from sklearn.ensemble import ExtraTreesClassifier


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

def check_list_in_lists(lst, set_of_lists):
    set_tuple_lists = {tuple(l) for l in set_of_lists}
    tuple_lst = tuple(lst)
    return tuple_lst in set_tuple_lists

iid = ['9034644']
# '9017909' einai o atlas

imageName = 'C:/Users/jaime/Desktop/all_kl0_registered_images/'+str(iid[0])+'/mri.hdr'
image = sitk.ReadImage(imageName)
maskName = 'C:/Users/jaime/Desktop/all_kl0_registered_images/'+str(iid[0])+'/mask.hdr'
mask = sitk.ReadImage(maskName)
    
paramsFile = 'C:/Users/jaime/YanAlgorithm/params-1.yaml'   
#a=sitk.RegionOfInterestImageFilter()
#a.SetRegionOfInterest([23,84,120,102,186,133]) # absolute ROI
#a.SetRegionOfInterest([18,79,115,107,191,138]) # slack ROI
#a.SetRegionOfInterest([70,180,180,30,40,40]) #9034644 / GOOD FOR TESTING PURPOSES WITH 9019287

#mask=a.Execute(mask)
#image=a.Execute(image)  

mask_arr = sitk.GetArrayFromImage(mask)

indices1= np.where(mask_arr == 1)
indices2= np.where(mask_arr == 2)
indices3= np.where(mask_arr == 3)

indices = random.sample(range(len(indices1[0])), 60000)
#indices.sort()
array1x = [indices1[0][i] for i in indices]
array1y = [indices1[1][i] for i in indices]
array1z = [indices1[2][i] for i in indices]

indices = random.sample(range(len(indices2[0])), 60000)
#indices.sort()
array2x = [indices2[0][i] for i in indices]
array2y = [indices2[1][i] for i in indices]
array2z = [indices2[2][i] for i in indices]

indices = random.sample(range(len(indices3[0])), 60000)
#indices.sort()
array3x = [indices3[0][i] for i in indices]
array3y = [indices3[1][i] for i in indices]
array3z = [indices3[2][i] for i in indices]
 
totindex = np.concatenate((array1x,array2x,array3x),axis=0)
totindey = np.concatenate((array1y,array2y,array3y),axis=0)
totindez = np.concatenate((array1z,array2z,array3z),axis=0)


count1 = 0
count2 = 0
count3 = 0

for i in range(0,len(totindex)):
    if mask_arr[totindex[i]][totindey[i]][totindez[i]] == 1:
        count1 +=1 
    if mask_arr[totindex[i]][totindey[i]][totindez[i]] == 2:
         count2 +=1 
    if mask_arr[totindex[i]][totindey[i]][totindez[i]] == 3:
        count3 +=1 
        
print(count1)
print(count2)
print(count3)

totindexx = list()

for i in range(0,len(totindex)):
    temp = list()
    temp.append(totindex[i])
    temp.append(totindey[i])
    temp.append(totindez[i])
    totindexx.append(temp)
 
set_of_lists = {tuple(l) for l in totindexx}

del totindex,totindey,totindez,array3x,array3y,array3z,array1x,array1y,array1z,indices,indices1,indices2,indices3


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


count = 0 
lbp_arr = np.zeros((len(lbp_arr1), len(lbp_arr1[0]), len(lbp_arr1[0][0])))
for i in range(0,len(lbp_arr1)):
    for j in range(0,len(lbp_arr1[0])):
        for k in range(0,len(lbp_arr1[0][0])):
            
            if lbp_arr1[i][j][k]!=0:
                lbp_arr[i][j][k]=lbp_arr1[i][j][k]
                if  lbp_arr31[i][j][k]!=0 or lbp_arr21[i][j][k]!=0 :
                    count +=1 
                    
            elif lbp_arr21[i][j][k]!=0:
                lbp_arr[i][j][k]=lbp_arr21[i][j][k]
                if  lbp_arr31[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 :
                    count +=1 
                    
            elif lbp_arr31[i][j][k]!=0:
                lbp_arr[i][j][k]=lbp_arr31[i][j][k]
                if  lbp_arr21[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 :
                    count +=1 
                

del L1,L2,L3,b1,b2,b3,c1,c2,c3,lbp_arr1,lbp_arr21,lbp_arr31,level1,level2,level21,level22,level31,level32   
                     
# TOTAL SAMPLING #    
counttot = 0    

for i in range(0,len(mask_arr)):
    for j in range(0,len(mask_arr[0])):
        for k in range(0,len(mask_arr[0][0])):
            temp = list()
            xyz = list()
            xyz = [i,j,k]
            lst = tuple(xyz)
            if mask_arr[i][j][k]!=0:
                if lst in set_of_lists:
                    counttot += 1
                else:
                    mask_arr[i][j][k]=0
    
print(counttot) 
mask = sitk.GetImageFromArray(mask_arr)
   
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


featureVector = extractor.execute(image, mask,label=1, voxelBased=True)
featureVector2 = extractor.execute(image, mask,label=2, voxelBased=True)
featureVector3 = extractor.execute(image, mask,label=3, voxelBased=True)

#kala feat glcm : Autocorrelation,jointAverage
#kala feat glrlm:  LongRunHighGrayLevelEmphasis,HighGrayLevelRunEmphasis,ShortRunHighGrayLevelEmphasis,
#kala gldm : grayLevelVariance(kaloutsiko),HighGrayLevelEmphasis
   
featlist = list()

#glcm: ['Autocorrelation','DifferenceEntropy','ClusterProminence','JointAverage','ClusterTendency','ClusterShade','Contrast','Correlation','DifferenceAverage','DifferenceVariance','JointEnergy','JointEntropy','SumSquares','Id','MaximumProbability','SumSquares','InverseVariance','SumEntropy','Imc1','Imc2','Idm','MCC','Idmn','Idn']

#ALL FEATURES
featlist = list()
featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Range','original_firstorder_MeanAbsoluteDeviation','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_Skewness','original_firstorder_Kurtosis','original_firstorder_Variance','original_firstorder_Uniformity'] #original_firstorder_StandardDeviation',
featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy']  #'original_glcm_JointAverage'
featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
featlist_gldm = ['original_gldm_SmallDependenceEmphasis','original_gldm_LargeDependenceEmphasis','original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized','original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis','original_glszm_LargeAreaLowGrayLevelEmphasis','original_glszm_LargeAreaHighGrayLevelEmphasis']

featlist.append(featlist_firstorder)
featlist.append(featlist_glcm)
featlist.append(featlist_glrlm)
featlist.append(featlist_gldm)   
featlist.append(featlist_ngtdm)
featlist.append(featlist_glszm)

all_dataset=list()
y=list()
for z in range(0,len(featlist)):
  
    dataset=list()
    features = list()
    c=0
    
    for q in featlist[z]:
        label1 = featureVector[q]
        label2 = featureVector2[q]
        label3 = featureVector3[q]
        Adder = sitk.AddImageFilter()
        Bidder=Adder.Execute(label1,label2)
        res=Adder.Execute(Bidder,label3)
        img_views=sitk.GetArrayViewFromImage(res)
        img_views = img_views.astype(np.float32)
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
                xyz = list()
                xyz = [i,j,k]
                lst = tuple(xyz)
                if z==0 and mask_view[i][j][k]!=0:
                    if lst in set_of_lists:
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
                if mask_view[i][j][k] != 0:
                    if lst in set_of_lists: #skipparoume mideniki klasi
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
   

all_features = list()
other_features = ['brightness','lbp','x','y','z']
#other_features = ['brightness','x','y','z']

all_features = other_features + featlist_firstorder + featlist_glcm + featlist_glrlm +featlist_gldm + featlist_ngtdm + featlist_glszm
#all_features =  featlist_firstorder + featlist_glcm + featlist_glrlm +featlist_gldm + featlist_ngtdm + featlist_glszm

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    
# Use RFE to select the top 3 features
rfe = RFE(clf, n_features_to_select=20)
X_new = rfe.fit_transform(X, y)

# Fit the classifier with the data
clf.fit(X_new, y)

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
best_feat = combined[combined[:,1].argsort()[::-1]]

'''
clf = RandomForestClassifier()

# Use RFE to select the top 3 features
rfe = RFE(clf, n_features_to_select=20)
X_new = rfe.fit_transform(X, y)

# Fit the classifier with the data
clf.fit(X_new, y)

# Obtain the feature mask
mask = rfe.support_

# Obtain the feature importance values
importance = clf.feature_importances_

# Print the results
#print("Selected features: ", iris.feature_names[mask])
#print("Feature importances: ", importance)

it = 0
for i in range(0,len(mask)):
    if mask[i] == True:
        print(all_features[i],importance[it]) 
        it += 1 

'''
