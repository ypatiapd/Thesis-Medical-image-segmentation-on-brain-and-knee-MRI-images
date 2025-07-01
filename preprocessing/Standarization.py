# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:56:58 2023

@author: ypatia
"""

import os
import tempfile
import shutil
import gc

import math

import numpy as np
import SimpleITK as sitk

from scipy.interpolate import interp1d
from sklearn.utils.extmath import cartesian
from skimage.exposure import rescale_intensity

#from ants import ants_image_io, ants_image
#from ants.utils import n3_bias_field_correction

from dipy.denoise import noise_estimate, nlmeans

#from numba import jit, prange
from joblib import Parallel, delayed, dump, load


class IntensityStandardizer:
    '''
    Nyul Intensity Standardization class

    Training phase:
        Learns histogram modes from set of training images

    Transformation phase:
        Piece-wise linearly matches histogram of test image to templated histogram

    ...

    Attributes
    ----------

    Methods
    -------

    '''

    def __init__(self, numPoints, sMin = 1, sMax = 100, pLow = 1, pHigh = 99):
        '''
        Initialize class variables
        '''

        self.numPoints = numPoints
        self.sMin = sMin
        self.sMax = sMax
        self.pLow = pLow
        self.pHigh = pHigh
        self.perc = np.asarray([pLow] + list(np.arange(0, 100, numPoints)[1:]) + [pHigh])
    

    def getLandmarks(self, image, mask):
        '''
        computes image landmarks
        '''

        image_nda = sitk.GetArrayFromImage(image)
        mask_nda = sitk.GetArrayFromImage(mask)

        lms = [np.percentile(image_nda[mask_nda > 0], i) for i in self.perc]
        mapping = interp1d([lms[0], lms[-1]], [self.sMin, self.sMax], fill_value = 'extrapolate')
        mapped_lms = mapping(lms)

        return mapped_lms


    def train(self, imageList, maskList):
        '''
        Computes landmarks for all images
        '''

        self.mean_landmarks = []
        mapped_landmarks = []

        for image_path, mask_path in zip(imageList, maskList):
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            mapped_lms = self.getLandmarks(image, mask)
            mapped_landmarks.append(mapped_lms)

        mapped_landmarks = np.asarray(mapped_landmarks)
        self.mean_landmarks = mapped_landmarks.mean(axis = 0)


    def transform(self, image, mask):
        '''
        Standardizes image histogram based on computed landmarks
        '''

        lms = self.getLandmarks(image, mask)
        mapping = interp1d(lms, self.mean_landmarks, fill_value = 'extrapolate')
        
        image_nda = sitk.GetArrayFromImage(image)
        mapped_image_nda = mapping(image_nda.ravel())
        mapped_image_nda = mapped_image_nda.reshape(image_nda.shape)
        
        mapped_image = sitk.GetImageFromArray(mapped_image_nda)
        mapped_image.CopyInformation(image)

        return mapped_image


    def saveModel(self, path):
        '''
        saves computed landmarks and model parameters
        '''

        model = {'pLow': self.pLow,
                 'pHigh': self.pHigh,
                 'sMin': self.sMin,
                 'sMax': self.sMax,
                 'landmarks': self.mean_landmarks}

        np.savez(path, model)


    def loadModel(self, path):
        '''
        loads computed landmarks and model parameters
        '''

        model = np.load(path, allow_pickle = True)
        return model
    
imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['01']
#n=['4']
counter=0
imageList = list()
maskList = list()

for ite in imgs:    
    imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'
    imageList.append(imageName)
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    maskList.append(maskName)
    counter=counter+1
    
standardizer = IntensityStandardizer(numPoints=100)
standardizer.train(imageList, maskList)
counter=0
for ite in imgs:  
    
    imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+ite+'.hdr'
    image = sitk.ReadImage(imageName)
    maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    mask = sitk.ReadImage(maskName)
    standardized_image = standardizer.transform(image,mask)
    sitk.WriteImage( standardized_image, 'C:/Users/ypatia/diplomatiki/stand_imgs/stand'+ite+'.hdr')
    counter=counter+1

