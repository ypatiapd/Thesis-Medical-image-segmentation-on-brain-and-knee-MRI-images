# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:29:23 2023

@author: ypatia
"""

import SimpleITK
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import mutual_info_score
import heapq


#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']

#imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']
imgs=['01','02','04','05','06','07','09','12','13','14','16','17','20','21','22','23','25','27','28','33','34','37','38','39']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']      

counter=0

images= list()
dists = list()
     
# Read in the first image3
#image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/hist_imgs/hist06.hdr')
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')
image1_array = sitk.GetArrayFromImage(image1)

# Create an empty list to store mutual information scores
scores = []

# Iterate through the list of images
for ite  in imgs:
    dists = list()
    # Read in the current image
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    image2_array = sitk.GetArrayFromImage(image2)
    
    # Compute the mutual information score between the two images
    score = mutual_info_score(image1_array.ravel(), image2_array.ravel())
    dists.append(score)
    dists.append(ite)
    images.append(dists)

# Sort the scores and corresponding image paths based on the scores
#scores, image_paths = zip(*sorted(zip(scores, image_paths), key=lambda x: x[0]))
sorted_list = list()
sorted_list = sorted(images, key=lambda x: x[0], reverse = True)
for i in range(0,2):        
    print(sorted_list[i])
'''top_10_indexes = heapq.nlargest(, range(len(scores)), scores.__getitem__)
print(top_10_indexes)
for i in range(0,len(top_10_indexes)):
    print(imgs[top_10_indexes[i]])'''