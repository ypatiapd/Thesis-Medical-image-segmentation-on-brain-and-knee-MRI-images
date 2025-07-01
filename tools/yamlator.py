# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:07:43 2022

@author: ypatia
"""

from __future__ import print_function

import logging
import os

import SimpleITK as sitk
import six

import radiomics
from radiomics import featureextractor, getFeatureClasses


def tqdmProgressbar():
  """
  This function will setup the progress bar exposed by the 'tqdm' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality
  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import tqdm
  radiomics.progressReporter = tqdm.tqdm


def clickProgressbar():
  """
  This function will setup the progress bar exposed by the 'click' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality.
  Because the signature used to instantiate a click progress bar is different from what PyRadiomics expects, we need to
  write a simple wrapper class to enable use of a click progress bar. In this case we only need to change the 'desc'
  keyword argument to a 'label' keyword argument.
  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  # Enable the GLCM class to show the progress bar
  extractor.enableFeatureClassByName('glcm')

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import click

  class progressWrapper:
    def __init__(self, iterable, desc=''):
      # For a click progressbar, the description must be provided in the 'label' keyword argument.
      self.bar = click.progressbar(iterable, label=desc)

    def __iter__(self):
      return self.bar.__iter__()  # Redirect to the __iter__ function of the click progressbar

    def __enter__(self):
      return self.bar.__enter__()  # Redirect to the __enter__ function of the click progressbar

    def __exit__(self, exc_type, exc_value, tb):
      return self.bar.__exit__(exc_type, exc_value, tb)  # Redirect to the __exit__ function of the click progressbar

  radiomics.progressReporter = progressWrapper

import numpy
import SimpleITK as sitk
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'


if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()

image = sitk.ReadImage(imageName)
mask1 = sitk.ReadImage(maskName)


#cropped mask gia grigora tests 
a=sitk.RegionOfInterestImageFilter()

#a.Image=image
a.SetRegionOfInterest([120,120,100,20,20,20])
#a.SetSize([5,5,5]) #size of ROI
#a.SetIndex([110,110,80])
#a.SetRegionOfInterest([110,110,80])

mask=a.Execute(mask1)

applyLog = False
applyWavelet = False

# Setting for the feature calculation.
# Currently, resampling is disabled.
# Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}

#
# If enabled, resample image (resampled image is automatically cropped.
#
interpolator = settings.get('interpolator')
resampledPixelSpacing = settings.get('resampledPixelSpacing')
if interpolator is not None and resampledPixelSpacing is not None:
  image, mask = imageoperations.resampleImage(image, mask, **settings)

bb, correctedMask = imageoperations.checkMask(image, mask,correctMask = "True")
if correctedMask is not None:
  mask = correctedMask
image, mask = imageoperations.cropToTumorMask(image, mask, bb)


paramsFile = 'C:/Users/ypatia/diplomatiki/params.yaml'

# Regulate verbosity with radiomics.verbosity
# radiomics.setVerbosity(logging.INFO)

# Get the PyRadiomics logger (default log-level = INFO
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize feature extractor using the settings file
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()

# Uncomment one of these functions to show how PyRadiomics can use the 'tqdm' or 'click' package to report progress when
# running in full python mode. Assumes the respective package is installed (not included in the requirements)

tqdmProgressbar()
# clickProgressbar()

print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
  if features is None or len(features) == 0:
    features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
  for f in features:
    print(f)
    print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

print("Calculating features")

all_features=numpy.zeros((20,20,20))

featureVector = extractor.execute(image, mask,label=1, voxelBased=True)
featureVector2 = extractor.execute(image, mask,label=2, voxelBased=True)
featureVector3 = extractor.execute(image, mask,label=3, voxelBased=True)
#paratiroume oti i eikona me tis times twn features pou epistrefetai exei max megethos oso i maska pou
#pername(ROI) , alla se periptwsi pou exei trigyrw pixels klasis 0 , epistrefei mikroteres diastaseis pinaka.
#otan pername oli ti maska, exei idies diastaseis me ti maska kai tin eikona ,diorthwmenes
a1=featureVector['original_firstorder_Mean']
a2=featureVector2['original_firstorder_Mean']
a3=featureVector3['original_firstorder_Mean']

'''a=featureVector['original_firstorder_StandardDeviation']
a=featureVector['log-sigma-1-0-mm-3D_firstorder_Mean']
a=featureVector['log-sigma-1-0-mm-3D_firstorder_StandardDeviation']
a=featureVector['log-sigma-3-0-mm-3D_firstorder_Mean']
a=featureVector['log-sigma-3-0-mm-3D_firstorder_StandardDeviation']
a=featureVector['wavelet-LLH_firstorder_Mean']
a=featureVector['wavelet-LLH_firstorder_StandardDeviation']'''

#des an ontws kroparei teis seires pou exoun mono midenika 

for i in range(0,20):
    for j in range(0,20):
        for z in range(0,20):
            if a1.GetPixel(i,j,z) !=0:
                all_features[i,j,z]=a1.GetPixel(i,j,z)
            elif  a2.GetPixel(i,j,z) !=0:           
                all_features[i,j,z]=a2.GetPixel(i,j,z)
            elif  a3.GetPixel(i,j,z) !=0:           
                all_features[i,j,z]=a3.GetPixel(i,j,z)
            #print(a.GetPixel(i,j,z))

for i in range(0,20):
    for j in range(0,20):
        for z in range(0,20):
            print(all_features[i,j,z])
            
'''for i in range(0,137):
    for j in range(0,171):
        for z in range(0,145):
            print(all_features[i,j,z])'''
            
'''for featureName, featureValue in six.iteritems(featureVector):
  if isinstance(featureValue, sitk.Image):
    sitk.WriteImage(featureValue, '%s_%s.nrrd' % (image, featureName))
    print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, image, featureName))
  else:
    print('%s: %s' % (featureName, featureValue))'''