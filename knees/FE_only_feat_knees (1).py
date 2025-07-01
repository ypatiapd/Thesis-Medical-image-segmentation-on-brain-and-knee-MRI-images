import logging
import os
import matplotlib.pyplot as plt

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
from skimage.feature import canny
import dipy 
import warnings
import cv2

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

def compute_euclidean_distance(mri_image):
    x, y, z = np.meshgrid(np.arange(mri_image.shape[0]), 
                          np.arange(mri_image.shape[1]), 
                          np.arange(mri_image.shape[2]))
    
    print(len(x))
    print(len(y))
    print(len(z))
    distances1 = np.sqrt((x - 0)**2 + (y - 78)**2 + (z - 58)**2)
    distances2 = np.sqrt((x - 157)**2 + (y - 78)**2 + (z - 58)**2)


    return distances1,distances2



def compute_edge_features(mri_image):
    edge_features = np.zeros_like(mri_image)

    for i in range(mri_image.shape[0]):
        slice_uint8 = cv2.normalize(mri_image[i, :, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Convert slice to uint8
        edges = cv2.Canny(slice_uint8, 50, 150)  # Compute edges in the i-th slice
        edge_features[i, :, :] = edges.astype(np.float32)  # Normalize edge values between 0 and 1

    for j in range(mri_image.shape[1]):
        slice_uint8 = cv2.normalize(mri_image[:, j, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Convert slice to uint8
        edges = cv2.Canny(slice_uint8, 50, 150)  # Compute edges in the j-th slice
        edge_features[:, j, :] += edges.astype(np.float32)  # Add edge values to the corresponding voxels

    for k in range(mri_image.shape[2]):
        slice_uint8 = cv2.normalize(mri_image[:, :, k], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Convert slice to uint8
        edges = cv2.Canny(slice_uint8, 50, 150)  # Compute edges in the k-th slice
        edge_features[:, :, k] += edges.astype(np.float32) # Add edge values to the corresponding voxels

    return edge_features

#iids = ['9011115','9017909','9019287','9023193', '9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761']#,'9073948']
iids = ['9011115','9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9073948','9089627','9090290','9093622','9080864']  #'9083500'  SERIOUSLY DAMAGED

iids = ['9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9073948','9089627','9090290','9093622','9080864']  #'9083500'  SERIOUSLY DAMAGED


for ite in iids:  
    
    imageName = 'C:/Users/jaime/Desktop/Normalized/normi'+str(ite)+'.hdr'
    maskName = 'C:/Users/jaime/Desktop/registered/'+str(ite)+'/mask.hdr'
    paramsFile = 'C:/Users/jaime/YanAlgorithm/params-1.yaml'   
   
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    
    a=sitk.RegionOfInterestImageFilter()
    a.SetRegionOfInterest([12,74,101,119,203,160]) # slack ROI for all Images except 9083500 in imgs :[106, 79, 17, 256, 272, 126]
    #a.SetRegionOfInterest([70,180,180,30,40,40]) #9034644 / GOOD FOR TESTING PURPOSES WITH 9019287
    
    mask=a.Execute(mask)
    image=a.Execute(image) 
    
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask)
    
    bbox = np.array(lsif.GetBoundingBox(1))
    bbox2 = np.array(lsif.GetBoundingBox(2))
    bbox3 = np.array(lsif.GetBoundingBox(3))
    bbox4 = np.array(lsif.GetBoundingBox(4))
    
    
    bboxs = np.vstack((bbox,bbox2))
    bboxs = np.vstack((bboxs,bbox3))
    bboxs = np.vstack((bboxs,bbox4))

    #print(bboxs)
    
    minx = 100000 # x OXI oti bazw sthn prwth 8esh tou maskArr
    miny = 100000 # y oti bazw sthn deuterh 8esh tou maskArr
    minz = 100000 # z OXI oti bazw sthn trith 8esh tou maskArr
    maxx = 0
    maxy = 0
    maxz = 0
    pminx,pminy,pminz = list(),list(),list()
    pmaxx,pmaxy,pmaxz = list(),list(),list()
    
    
    if np.all(bbox == bbox2) and np.all(bbox2 == bbox3) and np.all(bbox3 == bbox4) and np.all(bbox == bbox4):
        print("ALL GOOD")
    else:
        for i in range(0,3):
            
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
        maskArr[maxz][maxy][minx] = 4
        maskArr[minz][miny][maxx] = 4
        
        mask = sitk.GetImageFromArray(maskArr)
        
        image_arr = sitk.GetArrayFromImage(image)
        
        image.SetOrigin([0,0,0])
        

    image_arr = sitk.GetArrayFromImage(image)
    eucl1,eucl2 = compute_euclidean_distance(image_arr)  ###### EUCLIDEAN DISTANCES FROM PARALLEL SURFACES ######
    eucl1 = np.swapaxes(eucl1,0,1)
    eucl2 = np.swapaxes(eucl2, 0, 1)
    
    edge = compute_edge_features(image_arr)  ###### EDGE DETECTION ######     
    
    gradient_x = np.gradient(image_arr, axis=0)
    gradient_y = np.gradient(image_arr, axis=1)
    gradient_z = np.gradient(image_arr, axis=2)
    
    # Compute the magnitude and direction of the gradient for every voxel
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y) + np.square(gradient_z))
    gradient_direction = np.arctan2(np.sqrt(np.square(gradient_y) + np.square(gradient_z)), gradient_x)
    
    #gradient_direction = np.arctan2(np.sqrt(np.square(gradient_y) + np.square(gradient_z)), gradient_x)

    # Define the number of orientation bins
    n_bins = 8 
    n_binsB = 18
    # Define the range of the orientation histogram
    bin_range = (0, 2 * np.pi)
    
    # Compute the histogram for each voxel
    orientation_histogram = np.zeros((n_bins, gradient_direction.shape[0], gradient_direction.shape[1], gradient_direction.shape[2]))
    orientation_histogramB = np.zeros((n_binsB, gradient_direction.shape[0], gradient_direction.shape[1], gradient_direction.shape[2]))
    for i in range(gradient_direction.shape[0]):
        for j in range(gradient_direction.shape[1]):
            for k in range(gradient_direction.shape[2]):
                bin_number = np.digitize(gradient_direction[i, j, k], np.linspace(*bin_range, n_bins + 1))
                orientation_histogram[bin_number - 1, i, j, k] += gradient_magnitude[i, j, k]
                bin_numberB = np.digitize(gradient_direction[i, j, k], np.linspace(*bin_range, n_binsB + 1))
                orientation_histogramB[bin_numberB - 1, i, j, k] += gradient_magnitude[i, j, k]
    
    
    # Compute the mean orientation histogram for each voxel
    mean_orientation_histogram = np.mean(orientation_histogram, axis=0)
    mean_orientation_histogramB = np.mean(orientation_histogramB, axis=0)

    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=3)
    L4=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=2,lbp3DLevels=2,label=4)
     
    c1 = next(L1)
    level1= c1[0]   
    b1 =next(L1)
    level2= b1[0]
    a1 = next(L1)
    level1k = a1[0]
    lbp_arr1 = sitk.GetArrayFromImage(level1) 
    
    lbp_arr12 = sitk.GetArrayFromImage(level2) #second level 
    
    lbp_arr1k = sitk.GetArrayFromImage(level1k)  # kyrtosis   
    
    c2 = next(L2)
    level21= c2[0]
    b2 =next(L2)
    level22= b2[0] 
    a2 = next(L2)
    level2k = a2[0]
    lbp_arr21 = sitk.GetArrayFromImage(level21)
    
    lbp_arr22 = sitk.GetArrayFromImage(level22) #second level
    
    lbp_arr2k = sitk.GetArrayFromImage(level2k)  # kyrtosis   
    
    c3 = next(L3)
    level31= c3[0]
    b3 =next(L3)
    level32= b3[0] 
    a3 = next(L3)
    level3k = a3[0]
    lbp_arr31 = sitk.GetArrayFromImage(level31)
    
    lbp_arr32 = sitk.GetArrayFromImage(level32) #second level
    
    lbp_arr3k = sitk.GetArrayFromImage(level3k)  # kyrtosis   
    
    c4 = next(L4)
    level41 = c4[0]
    b4 = next(L4)
    level42 = b4[0]
    a4 = next(L4)
    level4k = a4[0]
    lbp_arr41 = sitk.GetArrayFromImage(level41)    
    
    lbp_arr42 = sitk.GetArrayFromImage(level42) #second level
    
    lbp_arr4k = sitk.GetArrayFromImage(level4k)  # kyrtosis   
    
    count1 = 0 
    count2 = 0 
    count3 = 0
    count4 = 0   
    
    count12 = 0 
    count22 = 0 
    count32 = 0
    count42 = 0
    
    count1k = 0 
    count2k = 0 
    count3k = 0
    count4k = 0 
    
    
    lbp_arr = np.zeros((len(lbp_arr1), len(lbp_arr1[0]), len(lbp_arr1[0][0])))
    for i in range(0,len(lbp_arr1)):
        for j in range(0,len(lbp_arr1[0])):
            for k in range(0,len(lbp_arr1[0][0])):
                
                if lbp_arr1[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr1[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr21[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count1 +=1 '''
                        
                elif lbp_arr21[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr21[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count2 +=1 '''
                        
                elif lbp_arr31[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr31[i][j][k]
                    '''if  lbp_arr21[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count3 +=1 '''
                    
                elif lbp_arr41[i][j][k]!=0:
                    lbp_arr[i][j][k]=lbp_arr41[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr21[i][j][k]!=0:
                        count4 +=1 '''
                        
    
    lbp_arr2 = np.zeros((len(lbp_arr12), len(lbp_arr12[0]), len(lbp_arr12[0][0])))
    for i in range(0,len(lbp_arr12)):
        for j in range(0,len(lbp_arr12[0])):
            for k in range(0,len(lbp_arr12[0][0])):
                
                if lbp_arr12[i][j][k]!=0:
                    lbp_arr2[i][j][k]=lbp_arr12[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr22[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count12 +=1 '''
                        
                elif lbp_arr22[i][j][k]!=0:
                    lbp_arr2[i][j][k]=lbp_arr22[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count22 +=1 '''
                        
                elif lbp_arr32[i][j][k]!=0:
                    lbp_arr2[i][j][k]=lbp_arr32[i][j][k]
                    '''if  lbp_arr22[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count32 +=1 '''
                    
                elif lbp_arr42[i][j][k]!=0:
                    lbp_arr2[i][j][k]=lbp_arr42[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr22[i][j][k]!=0:
                        count42 +=1 '''
    
    lbp_arrk = np.zeros((len(lbp_arr1k), len(lbp_arr1k[0]), len(lbp_arr1k[0][0])))
    for i in range(0,len(lbp_arr1k)):
        for j in range(0,len(lbp_arr1k[0])):
            for k in range(0,len(lbp_arr1k[0][0])):
                
                if lbp_arr1k[i][j][k]!=0:
                    lbp_arrk[i][j][k]=lbp_arr1k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr2k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count1k +=1 '''
                        
                elif lbp_arr2k[i][j][k]!=0:
                    lbp_arrk[i][j][k]=lbp_arr2k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count2k +=1 '''
                        
                elif lbp_arr3k[i][j][k]!=0:
                    lbp_arrk[i][j][k]=lbp_arr3k[i][j][k]
                    '''if  lbp_arr2k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count3k +=1 '''
                    
                elif lbp_arr4k[i][j][k]!=0:
                    lbp_arrk[i][j][k]=lbp_arr4k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr2k[i][j][k]!=0:
                        count4k +=1 '''
        
    print (count1,count2,count3,count4)    
    print (count12,count22,count32,count42)    
    print (count1k,count2k,count3k,count4k)
    
    del L1,L2,L3,L4
    del level1,level2,level1k
    del level21,level22,level2k
    del level31,level32,level3k
    del level41,level42,level4k
    del a1,a2,a3,a4
    del b1,b2,b3,b4
    del c1,c2,c3,c4
    del lbp_arr1,lbp_arr21,lbp_arr31,lbp_arr41
    del lbp_arr12,lbp_arr22,lbp_arr32,lbp_arr42
    del lbp_arr1k,lbp_arr2k,lbp_arr3k,lbp_arr4k
    
    
    L1=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=1)
    L2=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=2)
    L3=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=3)
    L4=imageoperations.getLBP3DImage(image,mask,voxelBased=True,binWidth=5,lbp3DIcosphereRadius=1,lbp3DLevels=2,label=4)
     
    c1 = next(L1)
    level1= c1[0]   
    b1 =next(L1)
    level2= b1[0]
    a1 = next(L1)
    level1k = a1[0]
    lbp_arr1 = sitk.GetArrayFromImage(level1) 
    
    lbp_arr12 = sitk.GetArrayFromImage(level2) #second level 
    
    lbp_arr1k = sitk.GetArrayFromImage(level1k)  # kyrtosis   
    
    c2 = next(L2)
    level21= c2[0]
    b2 =next(L2)
    level22= b2[0] 
    a2 = next(L2)
    level2k = a2[0]
    lbp_arr21 = sitk.GetArrayFromImage(level21)
    
    lbp_arr22 = sitk.GetArrayFromImage(level22) #second level
    
    lbp_arr2k = sitk.GetArrayFromImage(level2k)  # kyrtosis   
    
    c3 = next(L3)
    level31= c3[0]
    b3 =next(L3)
    level32= b3[0] 
    a3 = next(L3)
    level3k = a3[0]
    lbp_arr31 = sitk.GetArrayFromImage(level31)
    
    lbp_arr32 = sitk.GetArrayFromImage(level32) #second level
    
    lbp_arr3k = sitk.GetArrayFromImage(level3k)  # kyrtosis   
    
    c4 = next(L4)
    level41 = c4[0]
    b4 = next(L4)
    level42 = b4[0]
    a4 = next(L4)
    level4k = a4[0]
    lbp_arr41 = sitk.GetArrayFromImage(level41)    
    
    lbp_arr42 = sitk.GetArrayFromImage(level42) #second level
    
    lbp_arr4k = sitk.GetArrayFromImage(level4k)  # kyrtosis   
    
    count1 = 0 
    count2 = 0 
    count3 = 0
    count4 = 0   
    
    count12 = 0 
    count22 = 0 
    count32 = 0
    count42 = 0
    
    count1k = 0 
    count2k = 0 
    count3k = 0
    count4k = 0 
    
    
    lbp_arrS1 = np.zeros((len(lbp_arr1), len(lbp_arr1[0]), len(lbp_arr1[0][0])))
    for i in range(0,len(lbp_arr1)):
        for j in range(0,len(lbp_arr1[0])):
            for k in range(0,len(lbp_arr1[0][0])):
                
                if lbp_arr1[i][j][k]!=0:
                    lbp_arrS1[i][j][k]=lbp_arr1[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr21[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count1 +=1 '''
                        
                elif lbp_arr21[i][j][k]!=0:
                    lbp_arrS1[i][j][k]=lbp_arr21[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count2 +=1 '''
                        
                elif lbp_arr31[i][j][k]!=0:
                    lbp_arrS1[i][j][k]=lbp_arr31[i][j][k]
                    '''if  lbp_arr21[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr41[i][j][k]!=0:
                        count3 +=1 '''
                    
                elif lbp_arr41[i][j][k]!=0:
                    lbp_arrS1[i][j][k]=lbp_arr41[i][j][k]
                    '''if  lbp_arr31[i][j][k]!=0 or lbp_arr1[i][j][k]!=0 or lbp_arr21[i][j][k]!=0:
                        count4 +=1 '''
                        
    
    lbp_arrS2 = np.zeros((len(lbp_arr12), len(lbp_arr12[0]), len(lbp_arr12[0][0])))
    for i in range(0,len(lbp_arr12)):
        for j in range(0,len(lbp_arr12[0])):
            for k in range(0,len(lbp_arr12[0][0])):
                
                if lbp_arr12[i][j][k]!=0:
                    lbp_arrS2[i][j][k]=lbp_arr12[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr22[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count12 +=1 '''
                        
                elif lbp_arr22[i][j][k]!=0:
                    lbp_arrS2[i][j][k]=lbp_arr22[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count22 +=1 '''
                        
                elif lbp_arr32[i][j][k]!=0:
                    lbp_arrS2[i][j][k]=lbp_arr32[i][j][k]
                    '''if  lbp_arr22[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr42[i][j][k]!=0:
                        count32 +=1 '''
                    
                elif lbp_arr42[i][j][k]!=0:
                    lbp_arrS2[i][j][k]=lbp_arr42[i][j][k]
                    '''if  lbp_arr32[i][j][k]!=0 or lbp_arr12[i][j][k]!=0 or lbp_arr22[i][j][k]!=0:
                        count42 +=1 '''
    
    lbp_arrSk = np.zeros((len(lbp_arr1k), len(lbp_arr1k[0]), len(lbp_arr1k[0][0])))
    for i in range(0,len(lbp_arr1k)):
        for j in range(0,len(lbp_arr1k[0])):
            for k in range(0,len(lbp_arr1k[0][0])):
                
                if lbp_arr1k[i][j][k]!=0:
                    lbp_arrSk[i][j][k]=lbp_arr1k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr2k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count1k +=1 '''
                        
                elif lbp_arr2k[i][j][k]!=0:
                    lbp_arrSk[i][j][k]=lbp_arr2k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count2k +=1 '''
                        
                elif lbp_arr3k[i][j][k]!=0:
                    lbp_arrSk[i][j][k]=lbp_arr3k[i][j][k]
                    '''if  lbp_arr2k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr4k[i][j][k]!=0:
                        count3k +=1 '''
                    
                elif lbp_arr4k[i][j][k]!=0:
                    lbp_arrSk[i][j][k]=lbp_arr4k[i][j][k]
                    '''if  lbp_arr3k[i][j][k]!=0 or lbp_arr1k[i][j][k]!=0 or lbp_arr2k[i][j][k]!=0:
                        count4k +=1 '''
        
    print (count1,count2,count3,count4)    
    print (count12,count22,count32,count42)    
    print (count1k,count2k,count3k,count4k)
    
    del L1,L2,L3,L4
    del level1,level2,level1k
    del level21,level22,level2k
    del level31,level32,level3k
    del level41,level42,level4k
    del a1,a2,a3,a4
    del b1,b2,b3,b4
    del c1,c2,c3,c4
    del lbp_arr1,lbp_arr21,lbp_arr31,lbp_arr41
    del lbp_arr12,lbp_arr22,lbp_arr32,lbp_arr42
    del lbp_arr1k,lbp_arr2k,lbp_arr3k,lbp_arr4k
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
    featureVector4 = extractor.execute(image, mask,label=4, voxelBased=True)
    
    featlist = list()
    
    # BEST 30 COLLECTION WITH a.SetRegionOfInterest([80,155,215,45,70,80])
    '''
    featlist_firstorder = ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_InterquartileRange','original_firstorder_MeanAbsoluteDeviation']
    featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_Idmn','original_glcm_Idn']
    featlist_glrlm = ['original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_RunEntropy','original_glrlm_ShortRunHighGrayLevelEmphasis']
    featlist_gldm = ['original_gldm_LowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis']
    featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Complexity','original_ngtdm_Strength']
    featlist_glszm = ['original_glszm_ZonePercentage','original_glszm_SmallAreaHighGrayLevelEmphasis']
    '''
    
    featlist_firstorder = ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Median','original_firstorder_Energy']
    featlist_glcm = ['original_glcm_Id','original_glcm_Autocorrelation','original_glcm_Idmn','original_glcm_Idn','original_glcm_Idm']#,'original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_Correlation','original_glcm_SumEntropy']
    featlist_glrlm = ['original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_LowGrayLevelRunEmphasis']
       
    featlist.append(featlist_firstorder)
    featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)  
    
    
    pdf = pd.DataFrame()
        
    
    for z in range(0,len(featlist)):
    
        dataset = list()
        
        features = list()
        c=0
        
        for q in featlist[z]:
            label1=featureVector[q]
            label2=featureVector2[q]
            label3=featureVector3[q]
            label4 = featureVector4[q]
            Adder = sitk.AddImageFilter()
            Bidder=Adder.Execute(label1,label2)
            res2=Adder.Execute(Bidder,label3)
            res = Adder.Execute(res2,label4)
            img_views=sitk.GetArrayViewFromImage(res)
            features.append(copy.copy(img_views))
            c=c+1
    
    
        #normalize the features
        
        print("Ciao Ciaoo skoupidopaido Ypatia")
        
        dims = label1.GetSize()
        size = dims[2]*dims[1]*dims[0]
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
                        temp.append(eucl1[i][j][k])
                        temp.append(eucl2[i][j][k])
                        temp.append(edge[i][j][k])
                        temp.append(lbp_arr[i][j][k])
                        temp.append(lbp_arr2[i][j][k])
                        temp.append(lbp_arrk[i][j][k])
                        temp.append(lbp_arrS1[i][j][k])
                        temp.append(lbp_arrS2[i][j][k])
                        temp.append(lbp_arrSk[i][j][k])
                        temp.append(i)
                        temp.append(j)
                        temp.append(k)
                        temp.append(gradient_x[i][j][k])
                        temp.append(gradient_y[i][j][k])
                        temp.append(gradient_z[i][j][k])
                        temp.append(gradient_magnitude[i][j][k])
                        temp.append(mean_orientation_histogram[i][j][k])
                        temp.append(mean_orientation_histogramB[i][j][k])
                        for q in range(0,len(features)):
                            temp.append(features[q][i][j][k])
                        
                        #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                        #temp.append(dist)
                        if mask_view[i][j][k] != 0:
                            dataset.append(temp)    
                            
                        '''if not numpy.isnan(temp[0]):
                            dataset.append(temp)'''
        
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
            
            df_im = pd.DataFrame(dataset.loc[:,0:19].values,columns = [str(0), str(1) , str(2),str(3), str(4) , str(5),str(6),str(7),str(8),str(9),str(10),str(11),str(12),str(13),str(14),str(15),str(16),str(17),str(18),str(19)]) #stin teleutaia epanalipsi pairnw tis treis prwtes steiles pou einai oi klaseis, to brightness kai lbp 
            
            df2 = dataset.loc[:,20:len(featlist[z])+20].values #pairnw kai tis upoloipes pou einai ta xaraktiristika gia pca
          
            principalDf = pd.DataFrame(data = df2)
            #             , columns = [str(5*z+8), str(5*z+9), str(5*z+10),str(5*z+11), str(5*z+12)])#,str(7*z+10),str(7*z+11),str(7*z+12)])#,str(10*z+9),str(10*z+10),str(10*z+11)])
            
            pdf = pd.concat([pdf,principalDf], axis=1)
          
            pdf = pd.concat([df_im,pdf], axis=1) # prosthetw sto pdf ta pca components kai tis 3 prwtes stiles
            
            
    for i in range(1,len(pdf.columns)):
        pdf[pdf.columns[i]] = (pdf[pdf.columns[i]]-np.min(pdf[pdf.columns[i]]))/ (np.max(pdf[pdf.columns[i]])-np.min(pdf[pdf.columns[i]]))     
    
    
    #dataset = pd.DataFrame(dataset)
    print(len(pdf))
    pdf=pdf.dropna()
    print(len(dataset))
    
    
    with open('FOGLCRLM'+ite+'.txt', 'w') as output:
        for row in range(0,len(pdf)):
            s = ",".join(map(str, pdf.iloc[row]))
            output.write(str(s) + '\n')
            

    
    
    