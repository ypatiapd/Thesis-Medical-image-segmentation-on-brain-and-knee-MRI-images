# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:28:07 2022

@author: ypatia
"""


import numpy
import SimpleITK as sitk
import six


from sklearn import preprocessing
import numpy as np


from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'

if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()

image = sitk.ReadImage(imageName)
mask1 = sitk.ReadImage(maskName)



#v1=[[2,3,4],[4,5,6]]
#v2=


a=sitk.RegionOfInterestImageFilter()

#a.Image=image
a.SetRegionOfInterest([50,50,30,20,20,20])
#a.SetSize([5,5,5]) #size of ROI
#a.SetIndex([110,110,80])
#a.SetRegionOfInterest([110,110,80])

mask=a.Execute(mask1)


settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None,
            'voxelBased' :True
            }



# This is a non-active comment on a separate line
'''imageType:
    Original: {}
    LoG: {'sigma' : [1.0, 3.0]}  # This is a non active comment on a line with active code preceding it.
    Wavelet:
        binWidth: 10

featureClass:
    glcm:
    glrlm: []
    firstorder: ['Mean',
                 'StandardDeviation']
    shape:
        - Volume
        - SurfaceArea

setting:
    binWidth: 25
    resampledPixelSpacing:'''
        
        
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

#
# Show the first order feature calculations
#

'''settings={'kernelRadius':1,
       'maskedKernel':True,
       'voxelBatch':1}'''
    
firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)

#firstOrderFeatures.enableFeatureByName('Mean', True)
firstOrderFeatures.enableAllFeatures()

print('Will calculate the following first order features: ')
for f in firstOrderFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating first order features...')
results1 = firstOrderFeatures.execute()
print('done')

print('Calculated first order features: ')
for (key, val) in six.iteritems(results1):
  print('  ', key, ':', val)
