import SimpleITK as sitk
import matplotlib.pyplot as plt

import logging
import os

import SimpleITK as sitk
import six
import math
import pandas as pd
#import radiomics
#from radiomics import featureextractor, getFeatureClasses
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import time 
#import platipy
#from platipy.imaging.registration.utils import apply_transform

import numpy as np
import six

#from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

import dipy 
import warnings

import argparse
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk


import matplotlib.pyplot as plt
import pydicom

#iid = ['9019287','9033937','9034644','9036770','9036948','9040944','9041946','9047539','9052335','9069761']  # optimal vector ,'9023193','9036287' 

iid=['9019287','9034644','9023193','9033937','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761']#,'23','22','16','11','20']

iid=['9011115']
# '9017909' einai o atlas
#iid = ['9034644']

totalcube = list()
totminx = 1000
totminy = 1000
totminz = 1000
totmaxx = -10
totmaxy = -10
totmaxz = -10

for x in range(0,len(iid)):    
    '''
    minx = 0
    miny = 0
    minz = 0
    maxx = 0
    maxy = 0
    maxz = 0
    
    
    maxx2 = -10 
    maxy2 = -10
    maxz2 = -10 
    
    minx2 = 1000
    miny2 = 1000
    minz2 = 1000
    
    maxx4 = -10
    maxy4 = -10
    maxz4 = -10
    
    minx4 = 1000
    miny4 = 1000
    minz4 = 1000
    '''
    
    
    cube = list()
    cube2 = list()
    cube4 = list()

    maskName = 'D:/registered_imgs/9011115/mask.hdr'

    #maskName = 'C:/Users/jaime/Desktop/kl0_images_to_register/9017909/mask.hdr' # CHECKING THE ATLAS MINIMUM CUBE
    mask = sitk.ReadImage(maskName)
    mask_arr = sitk.GetArrayFromImage(mask)
    
    indices2 = np.where(mask_arr == 2)
    #print(len(indices2[0]))
    x_min, x_max = np.min(indices2[0]), np.max(indices2[0])
    y_min, y_max = np.min(indices2[1]), np.max(indices2[1])
    z_min, z_max = np.min(indices2[2]), np.max(indices2[2])
    cube2.append(x_min)
    cube2.append(y_min)
    cube2.append(z_min)
    cube2.append(x_max)
    cube2.append(y_max)
    cube2.append(z_max)
    
    indices4 = np.where(mask_arr == 4)
    #print(len(indices4[0]))
    x_min, x_max = np.min(indices4[0]), np.max(indices4[0])
    y_min, y_max = np.min(indices4[1]), np.max(indices4[1])
    z_min, z_max = np.min(indices4[2]), np.max(indices4[2])
    
    cube4.append(x_min)
    cube4.append(y_min)
    cube4.append(z_min)
    cube4.append(x_max)
    cube4.append(y_max)
    cube4.append(z_max)
    '''
    print('For the mask '+format(iid[x])+' the minimum cube for class 2 is:')
    print(cube2)    
    print('For the mask '+format(iid[x])+' the minimum cube for class 4 is:')
    print(cube4)      
    '''
    if cube2[0]<=cube4[0]:
        cube.append(cube2[0])
    else:
        cube.append(cube4[0])
    if cube2[1]<=cube4[1]:
        cube.append(cube2[1])
    else:
        cube.append(cube4[1])
    if cube2[2]<=cube4[2]:
        cube.append(cube2[2])
    else:
        cube.append(cube4[2])
    if cube2[3]>=cube4[3]:
        cube.append(cube2[3])
    else:
        cube.append(cube4[3])
    if cube2[4]>=cube4[4]:
        cube.append(cube2[4])
    else:
        cube.append(cube4[4])
    if cube2[5]>=cube4[5]:
        cube.append(cube2[5])
    else:
        cube.append(cube4[5]) 
    
    print('For the mask '+format(iid[x])+' the minimum cube for classes 2&4 is:')
    print(cube)
    
    if cube[0] <= totminx:
        totminx = cube[0]
        
    if cube[1] <= totminy:
        totminy = cube[1]
    
    if cube[2] <= totminz:
        totminz = cube[2]
        
    if cube[3] >= totmaxx:
        totmaxx = cube[3]
    
    if cube[4] >= totmaxy:
        totmaxy = cube[4]
        
    if cube[5] >= totmaxz:
        totmaxz = cube[5]
    
    
    '''
    for i in range(0,len(mask_arr)):
        for j in range(0,len(mask_arr[0])):
            for k in range(0,len(mask_arr[0][0])):
                
                if mask_arr[i][j][k] == 2:
                    if i > maxx2: 
                        maxx2 = i
                    if j > maxy2:
                        maxy2 = j
                    if k > maxz2:
                        maxz2 = k
                    if i < minx2:
                        minx2 = i
                    if j < miny2:
                        miny2 = j
                    if k < minz2:
                        minz2 = k
                if mask_arr[i][j][k] == 4:
                    if i > maxx4: 
                        maxx4 = i
                    if j > maxy4:
                        maxy4 = j
                    if k > maxz4:
                        maxz4 = k
                    if i < minx4:
                        minx4 = i
                    if j < miny4:
                        miny4 = j
                    if k < minz4:
                        minz4 = k
                
             
    if minx2 <= minx4:
        minx = minx2
    else:
        minx = minx4
    if miny2 <= miny4:
        miny = miny2
    else:
        miny = miny4 
    if minz2 <= minz4:
        minz = minz2
    else:
        minz = minz4
    if maxx2 >= maxx4:
        maxx = maxx2
    else:
        maxx = maxx4
    if maxy2 >= maxy4:
        maxy = maxy2
    else:
        maxy = maxy4
    if maxz2 >= maxz4:
        maxz = maxz2
    else:
        maxz = maxz4
    
    cube.append(minx)
    cube.append(miny)
    cube.append(minz)
    cube.append(maxx)
    cube.append(maxy)
    cube.append(maxz)
    
    cube2.append(minx2)
    cube2.append(miny2)
    cube2.append(minz2)
    cube2.append(maxx2)
    cube2.append(maxy2)
    cube2.append(maxz2)
    
    cube4.append(minx4)
    cube4.append(miny4)
    cube4.append(minz4)
    cube4.append(maxx4)
    cube4.append(maxy4)
    cube4.append(maxz4)
    
    print('For the mask '+format(iid[x])+' the minimum cube for classes 2&4 is:')
    print(cube)
    print('For the mask '+format(iid[x])+' the minimum cube for class 2 is:')
    print(cube2)    
    print('For the mask '+format(iid[x])+' the minimum cube for class 4 is:')
    print(cube4)      
    '''   
totalcube.append(totminx)
totalcube.append(totminy)
totalcube.append(totminz)
totalcube.append(totmaxx)
totalcube.append(totmaxy)
totalcube.append(totmaxz)
        
print('The minimum cube that encompasses classes 2&4 for all images is:')
print(totalcube)
        
# [107, 79, 20, 253, 272, 126]               