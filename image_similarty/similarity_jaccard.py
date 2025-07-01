# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:13:12 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:42:42 2023

@author: ypatia
"""

import SimpleITK
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import mutual_info_score
import heapq

import numpy as np

def jaccard_similarity(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    jaccard_index = intersection / union
    return jaccard_index

#imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']

imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']

counter=0
scores = list()
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')
image1_array = sitk.GetArrayFromImage(image1)

for ite in imgs:      
    
    # Read in the second image
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    image2_array = sitk.GetArrayFromImage(image2)
   
    # Compute the Euclidean distance between the two images
    score = jaccard_similarity(image1_array, image2_array)
    print(score)
    scores.append(score)
        
top_10_indexes = heapq.nlargest(10, range(len(scores)), scores.__getitem__)
print(top_10_indexes)
for i in range(0,len(top_10_indexes)):
    print(imgs[top_10_indexes[i]])
