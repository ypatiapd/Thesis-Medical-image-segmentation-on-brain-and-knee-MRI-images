# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:53:55 2022

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:48:16 2022

@author: jaime
"""

import logging
import os

import SimpleITK as sitk
import six
import math
import radiomics
from radiomics import featureextractor, getFeatureClasses

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

#imageName='C:/Users/jaime/YanAlgorithm/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'

paramsFile = 'C:/Users/jaime/YanAlgorithm/params.yaml'

'''if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()'''

#image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)


mask_view = sitk.GetArrayViewFromImage(mask)
limit1 = [0,22,44,66,88,110,132,154,176]
limit2 = [0,26,52,78,104,130,156,182,208]
limit3 = [0,22,44,66,88,110,132,154,176]

n=0

sum1 = 0
sum2 = 0
sum3 = 0

list1 = list()
list2 = list()
list3 = list()

list4 = list()
list5 = list()

x = -1

for z in range(0,8):
    for w in range(0,8):
        for q in range(0,8):       
            for i in range(limit1[z],limit1[z+1]):     
                for j in range(limit2[w],limit2[w+1]):
                    for k in range(limit3[q],limit3[q+1]):
                        if mask_view[i][j][k] == 1:
                            sum1= sum1+1
                        if mask_view[i][j][k] == 2:
                            sum2= sum2+1
                        if mask_view[i][j][k] == 3:
                            sum3= sum3+1
            list1.append(sum1)
            list2.append(sum2)
            list3.append(sum3) 
            
            list4.append(limit1[z])
            list4.append(limit2[w])
            list4.append(limit3[q])
            list5.append(list4)
            list4= list()

            sum1 = 0
            sum2 = 0
            sum3 = 0
        
for d in range(0,512):
    if list1[d] > 3800 and list2[d] > 3800 and list3[d] > 3800:
        print (list5[d])
        print ("Ponas?")
          
                
            
            
            
            
            
            
            
            
            
            
#print(get_size())


print("Ciao Ciaoo skoupidopaido Ypatia")