# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:35:08 2023

@author: jaime
"""

import os
import sys
import math

import numpy as np
import SimpleITK as sitk
'''
from Registration.utils import get_rigid_map, get_affine_map, get_bspline_map
from Registration.utils import get_binary_mask
from Registration.utils import sitk_to_numpy
'''
from utils import get_rigid_map, get_affine_map, get_bspline_map
from utils import get_binary_mask
from utils import sitk_to_numpy


#from ImagePreprocessing import histogram_matching
from ImagePreprocessing import rescale


def register(target_dir, atlas_dir, output_dir):
    '''
    Affinely register target image to atlas image &
    save transform parameters
    '''
    
    ### FIXING STAGE ###
    #read atlas mri
    atlas_mri = sitk.ReadImage(os.path.join(atlas_dir, 'mri.hdr'))
    
    #read target mri & mask
    target_id = os.path.split(target_dir)[-1]
    target_mri = sitk.ReadImage(os.path.join(target_dir, target_id, 'mri.hdr'))
    target_mask = sitk.ReadImage(os.path.join(target_dir, target_id, 'mask.hdr'))
    
    target_mask.CopyInformation(target_mri) 
    atlas_id = os.path.split(atlas_dir)[-1]
    target_id = os.path.split(target_dir)[-1]  # TO REMOVE IF ALL GOES WELL , SEEMS ABUNDANT
    
    
    

    #set parameter maps
    rmap = get_rigid_map()
    amap = get_affine_map()
    bmap = get_bspline_map()

    param_map = sitk.VectorOfParameterMap()
    #param_map.append(rmap) # COMMENT AFFINE TRANSFORMATION : SSD == 78000
    param_map.append(amap) # COMMENT AFFINE TRANSFORMATION : SSD == 38000
    #param_map.append(bmap) # COMMENT BSPLINE TRANSFORMATION

    #set elastix filter & transformix filters
    elx = sitk.ElastixImageFilter()
    trx = sitk.TransformixImageFilter()

    #set images  , RATHER OBVIOUS THAT ATLAS SHOULD BE FIXED AND TARGET MOVING
    elx.SetFixedImage(atlas_mri)
    elx.SetMovingImage(target_mri)
    elx.SetMovingMask(get_binary_mask(target_mask))

    #set parameter map
    elx.SetParameterMap(param_map)

    #set log to file
    elx.SetLogToConsole(False)
    elx.SetLogToFile(True)
    if not os.path.isdir(os.path.join(output_dir, atlas_id, target_id)):
        os.mkdir(os.path.join(output_dir, atlas_id, target_id))
    elx.SetOutputDirectory(os.path.join(output_dir, atlas_id, target_id))

    elx.SetNumberOfThreads(1)

    try:
        elx.Execute()
        registered_atlas = elx.GetResultImage()

        tmaps = elx.GetTransformParameterMap()
        for i in range(len(tmaps)):
            tmaps[i]['FinalBSplineInterpolationOrder'] = ['0']
            print(" THIS PRINT COUNTS THE NUMBER OF TMAMPS , PROPABLY ONE ITERATION AND NOT BSPLINE")
        
        trx.SetTransformParameterMap(tmaps)
        trx.SetMovingImage(target_mask)
        registered_mask = trx.Execute()
        registered_mask = sitk.Cast(registered_mask, sitk.sitkUInt8)
        
        
        #save registered image & mask as hdr files
        mri_path = os.path.join(output_dir, atlas_id, target_id, 'mri.hdr') # KINDA OBSOLETE , MAYBE USEFUL AFTER MODIFICATION
        print(mri_path)
        mask_path = os.path.join(output_dir, atlas_id, target_id, 'mask.hdr') # KINDA OBSOLETE , MAYBE USEFUL AFTER MODIFICATION
        sitk.WriteImage(registered_atlas, mri_path)
        sitk.WriteImage(registered_mask, mask_path)

        #return ssd
        print('Saved registered mri and mask')

        return 'Success'
    
    except Exception as error_msg:
        print(error_msg)
        return 'Failure'
        '''
        #save registered image & mask as NumPy Array
        mri_nda = sitk_to_numpy(registered_atlas)
        mask_nda = sitk_to_numpy(registered_mask)
        np.save(os.path.join(output_dir, target_id, atlas_id, 'mri.npy'), mri_nda)
        np.save(os.path.join(output_dir, target_id, atlas_id, 'mask.npy'), mask_nda)

        #ssd = evaluate_registration(target_mri, registered_atlas, registered_mask)

        #return ssd
        return 'Success'

    except Exception as error_msg:
        print(error_msg)
        return 'Failure'
        '''


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
    

    
def transform(target_id, source_dir, output_dir):
    '''
    Transform target mri & mask according to transformation parameters
    '''

    #Read atlas MRI & Mask
    #atlas_mri = sitk.ReadImage(os.path.join(source_dir, atlas_id, 'mri.mhd'))
    #atlas_mask = sitk.ReadImage(os.path.join(source_dir, atlas_id, 'mask.mhd'))
    
    # Read atlas MRI & Mask,our files are hdr
    target_mri = sitk.ReadImage(os.path.join(source_dir, target_id, 'mri.hdr'))
    target_mask = sitk.ReadImage(os.path.join(source_dir, target_id, 'mask.hdr'))
    print(source_dir)
    
    target_mask.CopyInformation(target_mri)   #sanity check

    #Read transform parameters
    param_files = os.listdir(os.path.join(output_dir, target_id))
    param_files.reverse()
    tmaps = [sitk.ReadParameterFile(os.path.join(output_dir, target_id, param_file)) for param_file in param_files]

    #Perform tranformation
    mri_trx = sitk.TransformixImageFilter()
    mri_trx.SetMovingImage(target_mri)
    for i in range(len(tmaps)):
        tmaps[i]['FinalBSplineInterpolationOrder'] = ['3']
    mri_trx.SetTransformParameterMap(tmaps)
    registered_mri = mri_trx.Execute()

    mask_trx = sitk.TransformixImageFilter()
    mask_trx.SetMovingImage(target_mask)
    for i in range(len(tmaps)):
        tmaps[i]['FinalBSplineInterpolationOrder'] = ['0']
    mask_trx.SetTransformParameterMap(tmaps)
    registered_mask = mask_trx.Execute()
    registered_mask = sitk.Cast(registered_mask, sitk.sitkUInt8)

    #rescale to [0, 100] for sanity check
    registered_mri = rescale(registered_mri)

    #convert to NumPy and save
    mri_nda = sitk_to_numpy(registered_mri)
    mask_nda = sitk_to_numpy(registered_mask)

    np.save(os.path.join(output_dir, target_id, 'mri.npy'), mri_nda)
    np.save(os.path.join(output_dir, target_id, 'mask.npy'), mask_nda)

'''

def dilate_average_mask(atlas_paths, image_dct = None):
    
    Construct average mask from selected_atlases and dilate
   

    #Average mask
    label_voting_filter = sitk.LabelVotingImageFilter()
    hole_filling_filter = sitk.BinaryFillholeImageFilter()
    hole_filling_filter.SetFullyConnected(True)
    
    masks_avg = []
    masks_dil = []
    for atlas_path in atlas_paths:
        mask_nda = np.load(os.path.join(atlas_path, 'mask.npy'))
        mask_avg = sitk.GetImageFromArray(mask_nda)
        masks_avg.append(mask_avg)

        #keep only cartilage labels
        mask_nda[mask_nda == 1] = 0
        mask_nda[mask_nda == 2] = 1
        mask_dil = sitk.GetImageFromArray(mask_nda)
        masks_dil.append(mask_dil)

    average_mask = label_voting_filter.Execute(masks_avg)
    average_mask = hole_filling_filter.Execute(average_mask)
    average_dil_mask = label_voting_filter.Execute(masks_dil)

    #Dilate to get sampling area
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius([50, 50, 50])
    dilated_mask = dilate_filter.Execute(average_dil_mask)

    return sitk.GetArrayFromImage(average_mask), sitk.GetArrayFromImage(dilated_mask)

'''

#iid = ['9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9011115']#
#iid=['9017909','9073948','9080864','9083500','9089627','9090290','9093622']
iid=['9017909','9019287']
#atlas_dir = 'C:/Users/jaime/Desktop/all_kl0_standardized_images/standard_knee'+str(iid[1])+'.hdr'
atlas_dir = 'D:/imgs2register/'+iid[0]+'/' # orizoume fakelo atlanta dhl fixed image
print('All images are bound to be registered on atlas '+format(iid[0]))

# 9023193 , 9036287 menoun ektos 

total_ssd = 0
 
for j in range(1,len(iid)):
    #target_dir = 'C:/Users/jaime/Desktop/all_kl0_standardized_images/standard_knee'+str(iid[j])+'.hdr'
    target_dir = 'D:/imgs2register/'+iid[j]+'/' # orizoume fakelo target eikonas dhl moving image
    output_dir = 'D:/registered_imgs/'+str(iid[j])+'/' # orizoume fakelo apo8hkeushs dhl registered image
    print('Current target image is set to be '+format(iid[j]))
    
    # Perform registration using the register function
    reg_result = register(target_dir, atlas_dir, output_dir)
    
    # Check if registration was successful
    if reg_result == 'Success':
        registered_mask = sitk.ReadImage(os.path.join(output_dir, 'mask.hdr'))
        registered_mri = sitk.ReadImage(os.path.join(output_dir, 'mri.hdr'))
        atlas_mri = sitk.ReadImage(os.path.join(atlas_dir, 'mri.hdr'))
        
    
        # Evaluate the registration performance using the evaluate_registration function
        ssd = evaluate_registration(atlas_mri, registered_mri, registered_mask)
        total_ssd = total_ssd + ssd
        # Print the registration performance
        print('Registration performance:', ssd)
    else:
        print('Registration failed')
        
print(total_ssd)