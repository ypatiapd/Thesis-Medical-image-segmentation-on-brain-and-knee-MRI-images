# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:51:54 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:56:18 2023

@author: ypatia
"""

import numpy as np
import SimpleITK as sitk
import heapq
from sklearn.metrics import mutual_info_score


# Read in the first image and its corresponding mask
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm33.hdr')
mask1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/disc1/OAS1_0033_MR1/FSL_SEG/OAS1_0033_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr')
# Convert the images and masks to arrays
image1_array = sitk.GetArrayFromImage(image1)
mask1_array = sitk.GetArrayFromImage(mask1)

counter=0
imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']      
scores=list()
for ite in imgs:
# Read in the second image and its corresponding mask
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    mask2= sitk.ReadImage('C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr')
    
    image2_array = sitk.GetArrayFromImage(image2)
    mask2_array = sitk.GetArrayFromImage(mask2)

    class2_mask1 = (mask1_array == 1)
    class2_mask2 = (mask2_array == 1)

    common_mask = np.logical_and(class2_mask1, class2_mask2)
    # Compute the Euclidean distance between the two images, taking into account the masks
    score = mutual_info_score(image1_array[common_mask].ravel(), image2_array[common_mask].ravel())
    scores.append(score)
    counter=counter+1

top_10_indexes = heapq.nlargest(12, range(len(scores)), scores.__getitem__)
print(top_10_indexes)
for i in range(0,len(top_10_indexes)):
    print(imgs[top_10_indexes[i]])