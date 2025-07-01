# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:29:40 2023

@author: ypatia
"""

import numpy as np

def dice_coefficient(img1, img2, threshold=30):
    # Threshold the images
    img1 = np.where(img1 > threshold, 0, 1)
    img2 = np.where(img2 > threshold, 0, 1)

    # Calculate the intersection and union of the two images
    intersection = np.sum(img1 * img2)
    union = np.sum(img1) + np.sum(img2)

    # Calculate the Dice coefficient
    dice = (2. * intersection) / union if union > 0 else 1.
    
    return dice

import SimpleITK
import SimpleITK as sitk
import numpy as np


#imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']

imgs=['01','02','04','05','06','07','09','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','32','33','34','37','38','39']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#xalia eikones  3,10,11,15,18,19,31,35,40,41,42
counter=0
images= list()
dists = list()
#image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm12.hdr')


for ite in imgs:      
    
    dists = list()
    # Read in the second image
    #image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')

    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)

    # Calculate the Dice coefficient similarity
    dice = dice_coefficient(image1_array, image2_array)

    print("Dice coefficient similarity:", dice)
    dists.append(dice)
    dists.append(ite)
    images.append(dists)


sorted_list = list()
sorted_list = sorted(images, key=lambda x: x[0],reverse = True)



