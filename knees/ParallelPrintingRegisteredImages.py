
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

iid = ['9011115','9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761']#,'9073948','9080864','9083500','9089627','9090290','9093622']   


for j in range(1,len(iid)-1):

    #histimageName = 'C:/Users/jaime/Desktop/all_kl0_standardized_images/standard_knee'+str(iid[1])+'.hdr'
    histimageName = 'C:/Users/jaime/Desktop/all_kl0_images_masks/mask'+str(iid[1])+'.hdr'
    histimage = sitk.ReadImage(histimageName)
    histimage_arr = sitk.GetArrayFromImage(histimage)
    #histimageName = 'C:/Users/jaime/Desktop/all_kl0_registered_images/'+str(iid[j+1])+'/mri.hdr'
    histimageName = 'C:/Users/jaime/Desktop/all_kl0_registered_images/'+str(iid[j+1])+'/mask.hdr'
    histimage = sitk.ReadImage(histimageName)
    histimage_arr2 = sitk.GetArrayFromImage(histimage)
        
    
    for i in range(0,int(len(histimage_arr[0][0])/5)):
        
        
        k = i*5
        fig, (ax3, ax4) = plt.subplots(1, 2,  figsize=(10, 5))
                
        ax3.imshow(histimage_arr[:,:,k], cmap='gray')
        ax3.set_title('Standard Image slice '+str(k)+'')
        
        ax4.imshow(histimage_arr2[:,:,k], cmap='gray')
        ax4.set_title('Registered Image slice '+str(k)+','+str(iid[j+1])+'')