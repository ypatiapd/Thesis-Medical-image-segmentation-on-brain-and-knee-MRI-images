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
    
    
iid = ['9011115','9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9073948','9080864','9083500','9089627','9090290','9093622']   
imageList = list()
maskList = list()

for j in range(0,len(iid)):
    
    imageName =  'C:/Users/jaime/Desktop/all_kl0_histogrammed_images/hist_knee'+str(iid[j])+'.hdr'
    imageList.append(imageName)
    maskName = 'C:/Users/jaime/Desktop/all_kl0_images_masks/mask'+str(iid[j])+'.hdr'
    maskList.append(maskName)
    
standardizer = IntensityStandardizer(numPoints=100)
standardizer.train(imageList, maskList)

for j in range(0,len(iid)):
    
    imageName =  'C:/Users/jaime/Desktop/all_kl0_histogrammed_images/hist_knee'+str(iid[j])+'.hdr'
    image = sitk.ReadImage(imageName)
    maskName = 'C:/Users/jaime/Desktop/all_kl0_images_masks/mask'+str(iid[j])+'.hdr'
    mask = sitk.ReadImage(maskName)
    standardized_image = standardizer.transform(image,mask)
    sitk.WriteImage(standardized_image, 'C:/Users/jaime/Desktop/all_kl0_standardized_images/standard_knee'+str(iid[j])+'.hdr')


