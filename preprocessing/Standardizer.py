# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:35:54 2023

@author: jaime
"""

"""
-------------------------
Data preprocessing script
-------------------------
"""

import os
import sys

import numpy as np
import SimpleITK as sitk

from multiprocessing import Pool
from functools import partial

from ImagePreprocessing import curvature_flow 
from ImagePreprocessing import bias_field_correction
from ImagePreprocessing import IntensityStandardizer
from ImagePreprocessing import median_filter


# Bias corrections filter parameters
downsample_factor = 3
n_iter = 10
time_step = 0.02

# Curvature flow filter parameters
numPoints = 10
smin = 1
smax = 100
plow = 1
phigh = 99


def BC_CF(read_subject_dir, write_dir, downsample_factor, n_iter, time_step):
    '''
    Perform N3 Bias Field Correction & Curvature Flow Filtering

    ...

    Parameters
    ----------
    read_subject_dit : str
        directory path containing MR images
    write_dir : str
        directory path to write filtered MR images
    downasmple_factor : list of ints
        downsample factor for each iteration of bias correction
    n_iter : int
        number of iterations
    time_step :
        time parameter of Curvature Flow filter

    '''

    reader = sitk.ImageSeriesReader()
    #read mri & mask
    for root, dirs, files in os.walk(read_subject_dir):
        if dirs == []:
            files.sort()
            filenames = [os.path.join(root, file) for file in files]
            reader.SetFileNames(filenames)
            mri = reader.Execute()
        if len(files) == 2:
            filename = [os.path.join(root, file) for file in files if file.endswith('.mhd')][0]
            mask = sitk.ReadImage(filename)
       
    #preprocess mri & manipulate mask
    mri = bias_field_correction(mri, downsample_factor = downsample_factor)
    mri = curvature_flow(mri, time_step = time_step, n_iter = n_iter)

    mask_nda = sitk.GetArrayFromImage(mask)
    mask_nda = np.flip(np.swapaxes(np.swapaxes(mask_nda, 0, 2), 1, 2), axis = 1)
    mask_nda[mask_nda == 4] = 2
    mask_nda[mask_nda == 3] = 1
    mask = sitk.GetImageFromArray(mask_nda)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(mri)

    #save to new folder
    subject_id = os.path.split(read_subject_dir)[-1]
    write_subject_dir = os.path.join(write_dir, subject_id)
    if not os.path.isdir(write_subject_dir):          
        os.mkdir(write_subject_dir)
    sitk.WriteImage(mri, os.path.join(write_subject_dir, 'mri.mhd'))
    sitk.WriteImage(mask, os.path.join(write_subject_dir, 'mask.mhd'))


def median(read_subject_dir, write_dir):
    '''
    Median filtering of MRI images

    ...

    Parameters
    ----------
    read_subject_dit : str
        directory path containing MR images
    write_dir : str
        directory path to write filtered MR images

    '''

    reader = sitk.ImageSeriesReader()
    #read mri & mask
    for root, dirs, files in os.walk(read_subject_dir):
        if dirs == []:
            files.sort()
            filenames = [os.path.join(root, file) for file in files]
            reader.SetFileNames(filenames)
            mri = reader.Execute()
        if len(files) == 2:
            filename = [os.path.join(root, file) for file in files if file.endswith('.mhd')][0]
            mask = sitk.ReadImage(filename)
   
    #apply median filter
    mri = median_filter(mri)

    #manipulate mask
    mask_nda = sitk.GetArrayFromImage(mask)
    mask_nda = np.flip(np.swapaxes(np.swapaxes(mask_nda, 0, 2), 1, 2), axis = 1)
    #mask_nda[mask_nda == 1] = 0
    #mask_nda[mask_nda == 3] = 0
    #mask_nda[mask_nda == 2] = 1
    #mask_nda[mask_nda == 4] = 2
   
    mask = sitk.GetImageFromArray(mask_nda)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(mri)

    #save to new folder
    subject_id = os.path.split(read_subject_dir)[-1]
    write_subject_dir = os.path.join(write_dir, subject_id)
    if not os.path.isdir(write_subject_dir):          
        os.mkdir(write_subject_dir)
    sitk.WriteImage(mri, os.path.join(write_subject_dir, 'mri.mhd'))
    sitk.WriteImage(mask, os.path.join(write_subject_dir, 'mask.mhd'))


def standardize_intensity(path, numPoints, smin, smax, plow, phigh):
    '''
    Standardize MRI instensity scale using Nyul normalization method
    Reference
    ...

    Parameters
    ----------
    path : str
        directory containing MR images
    numPoints : int
        number of histogram points to match
    smin : int
        low end of output intensity range
    smax : int
        high end of output intensity range
    plow : float
        low end of percentile range
    phigh : float
        high end of percentile range

    '''

    subjects = os.listdir(path)
    subjects.sort()

    standardizer = IntensityStandardizer(numPoints, sMin = smin, sMax = smax, pLow = plow, pHigh = phigh)
    imageList = [os.path.join(read_path, subject, 'mri.hdr') for subject in subjects]
    maskList = [os.path.join(read_path, subject, 'mask.hdr') for subject in subjects]
    
    print(imageList)

    #training phase
    standardizer.train(imageList, maskList)
    standardizer.saveModel(write_path)

    #transformation phase
    for subject in subjects:
        mri = sitk.ReadImage(os.path.join(read_path, subject, 'mri.hdr'))
        mask = sitk.ReadImage(os.path.join(read_path, subject, 'mask.hdr'))
        mapped = standardizer.transform(mri, mask)

        sitk.WriteImage(mapped, os.path.join(write_path, subject, 'mri.hdr'))
        sitk.WriteImage(mask, os.path.join(path, subject, 'mask.hdr'))
        
'''
#paths
read_path = sys.argv[1]
write_path = sys.argv[2]
if os.path.isdir(write_path):
    pass
else:
    os.mkdir(write_path)
'''

imgs=['01','02','04','05','06','07','09','11','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','31','32','33','34','37','38','39','40','42']
n=['4','4','4','4','4','3','4','4','4','4','4','3','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','3','3','4','4']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['11']#,'02','04','05','06','07']
#n=['4']#,'4','4','4','4','3']
counter=0;
for ite in imgs:    
    
    read_path ='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
 
    write_path = 'C:/Users/ypatia/diplomatiki/stand_imgs/stand'+ite+'.hdr'
 
    subjects = os.listdir(read_path)
    subjects.sort()
    subject_read_dirs = [os.path.join(read_path, subject) for subject in subjects]
 
 
    #Intensity standardization
    standardize_intensity(write_path, numPoints, smin, smax, plow, phigh)
    counter+=1
