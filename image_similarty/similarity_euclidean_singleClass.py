# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:56:18 2023

@author: ypatia
"""

import numpy as np
import SimpleITK as sitk
import heapq


# Read in the first image and its corresponding mask
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')
mask1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr')
# Convert the images and masks to arrays
image1_array = sitk.GetArrayFromImage(image1)
mask1_array = sitk.GetArrayFromImage(mask1)

counter=0
imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']
n=['4','4','4','4','3','4','4','4','3','3','4','4','3','4','4','4','3','3']#,'3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
distances=list()
for ite in imgs:
# Read in the second image and its corresponding mask
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    mask2= sitk.ReadImage('C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr')
    
    image2_array = sitk.GetArrayFromImage(image2)
    mask2_array = sitk.GetArrayFromImage(mask2)

    class2_mask1 = (mask1_array == 3)
    class2_mask2 = (mask2_array == 3)

    common_mask = np.logical_and(class2_mask1, class2_mask2)
    # Compute the Euclidean distance between the two images, taking into account the masks
    euclidean_distance = np.sqrt(np.sum((image1_array[common_mask] - image2_array[common_mask])**2))
    distances.append(euclidean_distance)
    print("Euclidean distance:", euclidean_distance)
    counter=counter+1

most_similar_idxs = heapq.nsmallest(4, range(len(distances)), distances.__getitem__)

similar_imgs=list()
for i in range(0,len(most_similar_idxs)):
    similar_imgs.append(imgs[most_similar_idxs[i]])

