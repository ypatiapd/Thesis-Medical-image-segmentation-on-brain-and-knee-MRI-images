
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


def histogram_matching(image, template, n_points = 6, n_levels = 100):
    '''
    Histogram matching image filter

    '''

    assert type(image) == sitk.SimpleITK.Image and type(template) == sitk.SimpleITK.Image, 'Incorrect image type'

    hist_match = sitk.HistogramMatchingImageFilter()
    hist_match.SetNumberOfMatchPoints(n_points)
    hist_match.SetNumberOfHistogramLevels(n_levels)

    image = hist_match.Execute(image, template)

    return image

iid = ['9011115','9017909','9019287','9023193','9033937','9034644','9036287','9036770','9036948','9040944','9041946','9047539','9052335','9069761','9073948','9080864','9083500','9089627','9090290','9093622']   


templateName =  'D:/median_imgs/median'+str(iid[0])+'.hdr'
template = sitk.ReadImage(templateName)

for j in range(1,len(iid)):
    imageName =  'D:/median_imgs/median'+str(iid[j])+'.hdr'
    image = sitk.ReadImage(imageName)
    
    hist_matched_image = histogram_matching(image, template)
    sitk.WriteImage(hist_matched_image, 'D:/hist_imgs/hist'+str(iid[j])+'.hdr')
