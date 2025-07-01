import os
import sys
import math

import numpy as np
import SimpleITK as sitk

from utils import get_rigid_map, get_affine_map, get_bspline_map
from utils import get_binary_mask
from utils import sitk_to_numpy


#from ImagePreprocessing import histogram_matching
from ImagePreprocessing import rescale


def evaluate_registration(atlas_mri, target_mri, target_mask):
    '''
    Evaluate registration performance on cartilage voxels
    '''

    #convert to numpy
    mask = sitk.GetArrayFromImage(target_mask)
    target_values = sitk.GetArrayFromImage(target_mri)[mask == 2]
    atlas_values = sitk.GetArrayFromImage(atlas_mri)[mask == 2]

    SSD = np.mean((atlas_values - target_values)**2)

    return SSD



filenames = ['9034644','9023193','9033937','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761']                                                         

filename = '9019287'
targetImageName = 'C:/Users/jaime/Desktop/Registered_Normalized/normi'+str(filename)+'.hdr'
target_image = sitk.ReadImage(targetImageName)

for i in filenames:
    optionImageName = 'C:/Users/jaime/Desktop/Registered_Normalized/normi'+str(i)+'.hdr'
    option_image = sitk.ReadImage(optionImageName)
    
    optionMaskName = 'C:/Users/jaime/Desktop/all_kl0_registered_images/'+str(i)+'/mask.hdr'
    option_mask = sitk.ReadImage(optionMaskName)
    
    # Evaluate the registration performance using the evaluate_registration function
    ssd = evaluate_registration(target_image, option_image, option_mask)
    # Print the registration performance
    print('Registration performance:'+ str(i)+'' , ssd )
