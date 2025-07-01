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
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
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
    
import numpy as np

def compute_corner_distances(image_array):
    # get shape of image array
    x, y, z = image_array.shape

    # create arrays for each corner
    corner_distances = [
        np.zeros((x, y, z)),  # top front left corner
        np.zeros((x, y, z)),  # top front right corner
        np.zeros((x, y, z)),  # top back left corner
        np.zeros((x, y, z)),  # top back right corner
        np.zeros((x, y, z)),  # bottom front left corner
        np.zeros((x, y, z)),  # bottom front right corner
        np.zeros((x, y, z)),  # bottom back left corner
        np.zeros((x, y, z))   # bottom back right corner
    ]

    # iterate over each voxel in the image array
    for i in range(x):
        for j in range(y):
            for k in range(z):
                # compute distances from each corner
                corner_distances[0][i,j,k] = np.sqrt(i**2 + j**2 + k**2)  # top front left corner
                corner_distances[1][i,j,k] = np.sqrt((x-i)**2 + j**2 + k**2)  # top front right corner
                corner_distances[2][i,j,k] = np.sqrt(i**2 + (y-j)**2 + k**2)  # top back left corner
                corner_distances[3][i,j,k] = np.sqrt((x-i)**2 + (y-j)**2 + k**2)  # top back right corner
                corner_distances[4][i,j,k] = np.sqrt(i**2 + j**2 + (z-k)**2)  # bottom front left corner
                corner_distances[5][i,j,k] = np.sqrt((x-i)**2 + j**2 + (z-k)**2)  # bottom front right corner
                corner_distances[6][i,j,k] = np.sqrt(i**2 + (y-j)**2 + (z-k)**2)  # bottom back left corner
                corner_distances[7][i,j,k] = np.sqrt((x-i)**2 + (y-j)**2 + (z-k)**2)  # bottom back right corner

    return corner_distances


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
iids = ['9011115','9019287','9036287','9036770','9069761','9041946','9073948','9090290','9023193']  


for ite in iids:   
    imageName = 'C:/Users/jaime/Desktop/registered/'+str(ite)+'/mri.hdr'
    maskName = 'C:/Users/jaime/Desktop/registered/'+str(ite)+'/mask.hdr'
    paramsFile = 'C:/Users/jaime/YanAlgorithm/params-1.yaml'   
   
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    
    a=sitk.RegionOfInterestImageFilter()
    #a.SetRegionOfInterest([12,74,101,119,203,160]) # slack ROI for all Images except 9083500 in imgs :[106, 79, 17, 256, 272, 126]
    #a.SetRegionOfInterest([70,180,180,30,40,40]) #9034644 / GOOD FOR TESTING PURPOSES WITH 9019287
    a.SetRegionOfInterest([16,80,122,127,204,154])
    
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
    gradient_xy = np.gradient(image_arr, axis=(0,1))
    gradient_xz = np.gradient(image_arr, axis=(0,2))
    gradient_yz = np.gradient(image_arr, axis=(1,2))
    gradient_xyz = np.gradient(image_arr) 
    
    
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
    
    # Compute the gradient orientation
    gradient_orientation = np.arctan2(np.sqrt(gradient_y**2 + gradient_x**2), gradient_z)
    
    # Convert the gradient orientation to degrees
    gradient_orientation_degrees = np.degrees(gradient_orientation)    
    
    distances = compute_corner_distances(image_arr)
    
    '''
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
    '''
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
    
    
    
    
    #featureVector = extractor.execute(image, mask,label=1, voxelBased=True)
    featureVector2 = extractor.execute(image, mask,label=2, voxelBased=True)
    #featureVector3 = extractor.execute(image, mask,label=3, voxelBased=True)
    featureVector4 = extractor.execute(image, mask,label=4, voxelBased=True)

    
    #kala feat glcm : Autocorrelation,jointAverage
    #kala feat glrlm:  LongRunHighGrayLevelEmphasis,HighGrayLevelRunEmphasis,ShortRunHighGrayLevelEmphasis,
    #kala gldm : grayLevelVariance(kaloutsiko),HighGrayLevelEmphasis
   
    featlist = list()
    
    
    #glcm: ['Autocorrelation','DifferenceEntropy','ClusterProminence','JointAverage','ClusterTendency','ClusterShade','Contrast','Correlation','DifferenceAverage','DifferenceVariance','JointEnergy','JointEntropy','SumSquares','Id','MaximumProbability','SumSquares','InverseVariance','SumEntropy','Imc1','Imc2','Idm','MCC','Idmn','Idn']

    #ALL FEATURES
    featlist = list()
    #featlist_firstorder= ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy','original_firstorder_Mean','original_firstorder_Maximum','original_firstorder_Minimum','original_firstorder_Median','original_firstorder_RootMeanSquared','original_firstorder_TotalEnergy','original_firstorder_Entropy','original_firstorder_InterquartileRange','original_firstorder_Range','original_firstorder_MeanAbsoluteDeviation','original_firstorder_RobustMeanAbsoluteDeviation','original_firstorder_Skewness','original_firstorder_Kurtosis','original_firstorder_Variance','original_firstorder_Uniformity'] #original_firstorder_StandardDeviation',
    #featlist_glcm = ['original_glcm_Autocorrelation','original_glcm_DifferenceEntropy','original_glcm_ClusterProminence','original_glcm_ClusterTendency','original_glcm_ClusterShade','original_glcm_Contrast','original_glcm_DifferenceAverage','original_glcm_DifferenceVariance','original_glcm_JointEnergy','original_glcm_JointEntropy','original_glcm_SumSquares','original_glcm_Correlation','original_glcm_Id','original_glcm_Idm','original_glcm_Idmn','original_glcm_Idn','original_glcm_Imc1','original_glcm_Imc2','original_glcm_InverseVariance','original_glcm_MaximumProbability','original_glcm_MCC','original_glcm_SumEntropy']  #'original_glcm_JointAverage'
    featlist_glrlm = ['original_glrlm_LongRunHighGrayLevelEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_GrayLevelNonUniformityNormalized','original_glrlm_RunLengthNonUniformity','original_glrlm_RunLengthNonUniformityNormalized','original_glrlm_RunPercentage','original_glrlm_GrayLevelVariance','original_glrlm_RunVariance','original_glrlm_RunEntropy','original_glrlm_LowGrayLevelRunEmphasis','original_glrlm_HighGrayLevelRunEmphasis','original_glrlm_ShortRunLowGrayLevelEmphasis','original_glrlm_ShortRunHighGrayLevelEmphasis','original_glrlm_LongRunLowGrayLevelEmphasis','original_glrlm_LongRunHighGrayLevelEmphasis']
    #featlist_gldm = ['original_gldm_SmallDependenceEmphasis','original_gldm_LargeDependenceEmphasis','original_gldm_GrayLevelNonUniformity','original_gldm_DependenceNonUniformity','original_gldm_DependenceNonUniformityNormalized','original_gldm_GrayLevelVariance','original_gldm_DependenceVariance','original_gldm_DependenceEntropy','original_gldm_LowGrayLevelEmphasis','original_gldm_HighGrayLevelEmphasis','original_gldm_SmallDependenceLowGrayLevelEmphasis','original_gldm_SmallDependenceHighGrayLevelEmphasis','original_gldm_LargeDependenceLowGrayLevelEmphasis','original_gldm_LargeDependenceHighGrayLevelEmphasis']
    #featlist_ngtdm = ['original_ngtdm_Coarseness','original_ngtdm_Contrast','original_ngtdm_Busyness','original_ngtdm_Complexity','original_ngtdm_Strength']
    #featlist_glszm = ['original_glszm_SmallAreaEmphasis','original_glszm_LargeAreaEmphasis','original_glszm_GrayLevelNonUniformity','original_glszm_GrayLevelNonUniformityNormalized','original_glszm_SizeZoneNonUniformity','original_glszm_SizeZoneNonUniformityNormalized','original_glszm_ZonePercentage','original_glszm_GrayLevelVariance','original_glszm_ZoneVariance','original_glszm_ZoneEntropy','original_glszm_LowGrayLevelZoneEmphasis','original_glszm_HighGrayLevelZoneEmphasis','original_glszm_SmallAreaLowGrayLevelEmphasis','original_glszm_SmallAreaHighGrayLevelEmphasis','original_glszm_LargeAreaLowGrayLevelEmphasis','original_glszm_LargeAreaHighGrayLevelEmphasis']
    
    
    #featlist.append(featlist_firstorder)
    #featlist.append(featlist_glcm)
    featlist.append(featlist_glrlm)
    #featlist.append(featlist_gldm)   
    #featlist.append(featlist_ngtdm)
    #featlist.append(featlist_glszm)
    
    
    all_dataset=list()
    y=list()
    for z in range(0,len(featlist)):
      
        dataset=list()
        features = list()
        c=0
        
        for q in featlist[z]:
            #label1=featureVector[q]
            label2=featureVector2[q]
            #label3=featureVector3[q]
            label4 = featureVector4[q]
            Adder = sitk.AddImageFilter()
            Bidder=Adder.Execute(label4,label2)
            #res2=Adder.Execute(Bidder,label3)
            #res = Adder.Execute(res2,label4)
            img_views=sitk.GetArrayViewFromImage(Bidder)
            features.append(copy.copy(img_views))
            c=c+1
    
        
        #normalize the features
        
        print("Ciao Ciaoo skoupidopaido Ypatia")
        
        #dims=[30,30,30]
        dims = label2.GetSize()
        size = dims[2]*dims[1]*dims[0]
        #gdoup = numpy.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
        del label2,Adder,Bidder
    
        #for i in range(0,len(features)):
        #    features[i] = (features[i]-numpy.min(features[i]))/ (numpy.max(features[i])-numpy.min(features[i]))
        mask_view = sitk.GetArrayFromImage(mask)
        bright_view = sitk.GetArrayFromImage(image)
        
       
        for i in range(0,dims[2]):
            for j in range(0,dims[1]):
                for k in range(0,dims[0]):
                    temp = list()
                    if z==0 and mask_view[i][j][k]!=0 and mask_view[i][j][k]!=1 and mask_view[i][j][k]!=3:
                        y.append(mask_view[i][j][k])
                    if z==0:
                        temp = list()
                        #temp.append(mask_view[i][j][k])
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
                        temp.append(gradient_xy[0][i,j,k])
                        temp.append(gradient_xy[1][i,j,k])
                        temp.append(gradient_xz[0][i,j,k])
                        temp.append(gradient_xz[1][i,j,k])
                        temp.append(gradient_yz[0][i,j,k])
                        temp.append(gradient_yz[1][i,j,k])
                        temp.append(gradient_xyz[0][i,j,k])
                        temp.append(gradient_xyz[1][i,j,k])
                        temp.append(gradient_xyz[2][i,j,k])                        
                        temp.append(gradient_magnitude[i][j][k])
                        temp.append(gradient_orientation[i][j][k])
                        temp.append(distances[0][i,j,k])
                        temp.append(distances[1][i,j,k])
                        temp.append(distances[2][i,j,k])
                        temp.append(distances[3][i,j,k])
                        temp.append(distances[4][i,j,k])
                        temp.append(distances[5][i,j,k])
                        temp.append(distances[6][i,j,k])
                        temp.append(distances[7][i,j,k])

                    #temp.append(mask_view[i][j][k]) #pairnoume ti maska gia na skiparoume mideniki klasi sto dataset
                    for q in range(0,len(features)):
                        temp.append(features[q][i][j][k])
                    #temp.append(bright_view[i][j][k])
                    #dist= math.sqrt(pow(i,2)+pow(j,2)+pow(k,2))
                    #temp.append(dist)
                    if mask_view[i][j][k] != 0 and mask_view[i][j][k]!=1 and mask_view[i][j][k]!=3: #skipparoume mideniki klasi
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
    other_features = ['brightness','eucl1','eucl2','edge','lbp1','lbp2','lbpk','lbps1','lbps2','lbpsk','x','y','z','gradx','grady','gradz','gradxy1','gradxy2','gradxz1','gradxz2','gradyz1','gradyz2','gradxyz1','gradxyz2','gradxyz3','magnitude','orientation','distance1','distance2','distance3','distance4','distance5','distance6','distance7','distance8']
    #all_features = other_features + featlist_firstorder + featlist_glcm + featlist_glrlm +featlist_gldm + featlist_ngtdm + featlist_glszm
    all_features = other_features + featlist_glrlm 
    
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    
    # Use RFE to select the top 3 features
    rfe = RFE(clf, n_features_to_select=30)
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
        
    '''clf = RandomForestClassifier()

    # Use RFE to select the top 3 features
    rfe = RFE(clf, n_features_to_select=30)
    X_new = rfe.fit_transform(X, Y)
    
    # Fit the classifier with the data
    clf.fit(X_new, Y)
    
    # Obtain the feature mask
    mask = rfe.support_
    
    # Obtain the feature importance values
    importance = clf.feature_importances_
    
    # Print the results
    #print("Selected features: ", iris.feature_names[mask])
    print("Feature importances: ", importance)
    
    for i in range(0,len(mask)):
        if mask[i] == True:
            print(all_features[i],importance[i])'''
    
    
    
    

    
    '''
    clf = RandomForestClassifier()

    # Create the RFE selector and set the number of features to select
    rfe = RFE(clf, n_features_to_select=30)
    
    # Fit the RFE selector to the data
    rfe.fit(X, y)
    
    # Get the boolean mask of selected features
    mask = rfe.support_
    
    # Get the feature importance values of the selected features
    importance = clf.feature_importances_
    important_features = importance[mask]  
    
    # Create an instance of SVC
    estimator = SVC(kernel="linear")
    
    # Create an instance of RFE
    selector = RFE(estimator, n_features_to_select=30)
    
    # Fit the RFE to the data
    selector = selector.fit(X, y)
    
    # Print the selected features
    #print(selector.support_)
    
    #print(selector.ranking_)
    
    for i in range(0,len(all_features)):
        if selector.support_[i] == True :
            print (all_features[i])#,selector.ranking_[i])         
  
    # Create an instance of SVC
    estimator = SVC(kernel="linear")
    
    # Create an instance of RFE
    selector = RFE(estimator, n_features_to_select=20)
    
    # Fit the RFE to the data
    selector = selector.fit(X, y)
    
    # Print the selected features
    #print(selector.support_)
    
    #print(selector.ranking_)
    
    for i in range(0,len(all_features)):
        if selector.support_[i] == True :
            print (all_features[i])#,selector.ranking_[i])
           
    estimator = SVC(kernel="linear")
    
    # Create an instance of RFE
    selector = RFE(estimator, n_features_to_select=10)
    
    # Fit the RFE to the data
    selector = selector.fit(X, y)
    
    # Print the selected features
    #print(selector.support_)
    
    #print(selector.ranking_)
    
    for i in range(0,len(all_features)):
        if selector.support_[i] == True :
            print (all_features[i])#,selector.ranking_[i])
          
            
            
    
   
    with open('Most_important_knees.txt', 'w') as f:
       for item in selector.support_:
           f.write(str(item) + '\n')
       
    # Create an instance of RFE
    selector = RFE(estimator, n_features_to_select=30)
    
    # Fit the RFE to the data
    selector = selector.fit(X, y)
    
    # Print the selected features
    print(selector.support_)
    
    print(selector.ranking_)'''




