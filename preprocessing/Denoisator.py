# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:07:47 2022

@author: jaime
"""


from __future__ import print_function

import numpy
import SimpleITK as sitk
import six
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

#7,15,16,20,26,34,38,39 n3
#imgs=['13','28','33','23','22','16','11','20']
#n=['3','3','4','3',
#n=['4','4','4','4','4','3','4','3']
#imgs=['38','25','05','26','07','29','06','39','09']
#n=['3','4','4','3','3','4','4','3','4']
imgs=['01','02','04','05','06','07','09','11','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','31','32','33','34','37','38','39','40','42']
n=['4','4','4','4','4','3','4','4','4','4','4','3','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','3','3','4','4']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['11']#,'02','04','05','06','07']
#n=['4']#,'4','4','4','4','3']
counter=0;
for ite in imgs:    
    imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+ite+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111_brain.nii'
    maskName= 'D:/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'

    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    #maskName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+ite+'_MR1/FSL_SEG/OAS1_00'+ite+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    paramsFile = 'C:/Users/ypatia/diplomatiki/params.yaml'
    
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    
    data = sitk.GetArrayFromImage(image)
    
    sigma = estimate_sigma(data, N=16)
    
    nlmimage = nlmeans(data, sigma=sigma, patch_radius=1,
                  block_radius=2, rician=False)
    
    image = sitk.GetImageFromArray(nlmimage)
    
    sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr')
    
    counter+=1