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
import numpy as np
import six
import dipy 
import warnings

import argparse
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from intensity_normalization.normalize.nyul import NyulNormalize
import matplotlib.pyplot as plt
import pydicom

# '9017909',
iid = ['9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9011115','9017909','9073948','9080864','9083500','9089627','9090290','9093622']
#iid = ['9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761']#,'9073948','9080864','9083500','9089627','9090290','9093622']   
image_paths = list()

for z in iid:    
    #imageName='C:/Users/jaime/Desktop/DenoiKnees/denoi'+str(z)+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    imageName =  'D:/registered_imgs/'+str(z)+'/mri.hdr'
    image_paths.append(imageName)
    images = [nib.load(image_path).get_fdata() for image_path in image_paths]
    
    # normalize the images and save the standard histogram
    
    nyul_normalizer = NyulNormalize()
    nyul_normalizer.fit(images)
    normalized = [nyul_normalizer(image) for image in images]
    nyul_normalizer.save_standard_histogram("standard_histogram.npy")
    
    for i in range(0,len(normalized)):
        image= sitk.GetImageFromArray(normalized[i])
        image_arr = sitk.GetArrayFromImage(image)
    
        image_arr = np.swapaxes(image_arr,0,2)
        image=sitk.GetImageFromArray(image_arr)
        #sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/norm_imgs/norm'+imgs[i]+'.hdr')
        sitk.WriteImage(image, 'D:/norm_imgs/norm'+str(z)+'.hdr')
    