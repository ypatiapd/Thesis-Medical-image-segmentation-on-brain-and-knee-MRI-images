# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:22:31 2023

@author: ypatia
"""

from skimage.feature import greycomatrix, greycoprops
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
    
    distances = [1]  # distances for greycomatrix calculation
    angles = [0, 1, 2]  # angles for greycomatrix calculation
    levels = 256  # number of intensity levels
    symmetric = True  # whether the GLCM is symmetric or not
    normed = True  # whether to normalize the GLCM or not
    
    # Calculate the grey-level co-occurrence matrices for both images
    g1 = greycomatrix(image1_array, distances=distances, angles=angles, levels=levels,
                      symmetric=symmetric, normed=normed)
    g2 = greycomatrix(image2_array, distances=distances, angles=angles, levels=levels,
                      symmetric=symmetric, normed=normed)
    
    # Calculate the texture correlation coefficient
    tcc = np.corrcoef(g1.ravel(), g2.ravel())[0, 1]
    # Compute the Euclidean distance between the two images
    
    dists.append(tcc)
    dists.append(ite)
    images.append(dists)
    
    
    
    print("tcc:", tcc)

sorted_list = list()
sorted_list = sorted(images, key=lambda x: x[0])
