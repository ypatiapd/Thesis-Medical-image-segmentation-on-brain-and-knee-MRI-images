# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:08:12 2023

@author: ypatia
"""





import SimpleITK
import SimpleITK as sitk
import numpy as np

def correlation(arr1, arr2):
    # Subtract the mean of each array
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    return np.dot(arr1, arr2) / len(arr1)


#imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']

imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']

counter=0
images= list()
dists = list()
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm33.hdr')


for ite in imgs:      
    
    dists = list()
    # Read in the second image
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')

    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)
    
    
    # Compute the Euclidean distance between the two images
    corr = correlation(image1_array , image2_array)
    dists.append(corr)
    dists.append(ite)
    images.append(dists)
    
    
    
    print("correlation:", corr)

sorted_list = list()
sorted_list = sorted(images, key=lambda x: x[0],reverse=True)
