# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:39:34 2022

@author: ypatia
"""

import numpy
import SimpleITK as sitk
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

# testBinWidth = 25 this is the default bin size
# testResampledPixelSpacing = [3,3,3] no resampling for now.

# Get some test data

# Download the test case to temporary files and return it's location. If already downloaded, it is not downloaded again,
# but it's location is still returned.
imageName, maskName = getTestCase('brain1')
num='01'

mean=[]
var=[]
ptr=[]
#7,15,16,20,24,26,34,36,38,39
for i in range(1,10):
    ptr.append('0'+format(i))
for i in range(10,43):
    ptr.append(format(i))

for i in range(0,len(ptr)):
    try:
        imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ptr[i]+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ptr[i]+'_MR1_mpr_n4_anon_sbj_111.hdr'
        maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ptr[i]+'_MR1/FSL_SEG/OAS1_00'+ptr[i]+'_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
        
        if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
          print('Error getting testcase!')
          exit()
        
        image = sitk.ReadImage(imageName)
        mask = sitk.ReadImage(maskName)
        
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
        
        
        #
        # Show the first order feature calculations
        #
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
        
        #firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableAllFeatures()
        
        print('Will calculate the following first order features: ')
        for f in firstOrderFeatures.enabledFeatures.keys():
          print('  ', f)
          print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)
        
        print('Calculating first order features...')
        results = firstOrderFeatures.execute()
        print('done')
        mean.append(results['Mean'])
        var.append(results['Variance'])
        
        print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
          print('  ', key, ':', val)
      
    except RuntimeError:
        print('excepttt')
        #continue
        
diff=[]
min_ptr=[]
min_diffs=[]

for i in range(1,len(mean)):
    diff.append(abs(mean[0]-mean[i]))
    
p=numpy.percentile(diff,33)

for i in range(0,len(mean)-1):
    if diff[i]<=p:
        min_ptr.append(i)
        min_diffs.append(diff[i])
    
        


