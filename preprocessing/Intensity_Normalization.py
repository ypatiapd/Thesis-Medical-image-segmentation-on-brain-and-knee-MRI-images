# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:59:03 2022

@author: jaime
"""
import SimpleITK as sitk
import nibabel as nib
import numpy as np

imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#imgs=['01']
#n=['4']

#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#main_path='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
#imgs=['38','25','05','26','07','29','06','39','09']
#imgs=['13','28','33','23','22','16','11','20']
#n=['3','3','4','3',
#n=['4','4','4','4','4','3','4','3']
#imgs=['01','02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']
#n=['4','3','4','4','3','3','4','3','4','3']
counter=0;
image_paths = list()
for z in imgs:
    #imageName='C:/Users/ypatia/diplomatiki/registered_imgs/registered'+z+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/hist_imgs/hist'+z+'.hdr'

    imageName='C:/Users/ypatia/diplomatiki/denoised/denoised'+z+'.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    image_paths.append(imageName)
    counter=counter+1
images = [nib.load(image_path).get_fdata() for image_path in image_paths]

# normalize the images and save the standard histogram
from intensity_normalization.normalize.nyul import NyulNormalize

nyul_normalizer = NyulNormalize()
nyul_normalizer.fit(images)
normalized = [nyul_normalizer(image) for image in images]
nyul_normalizer.save_standard_histogram("standard_histogram.npy")

for i in range(0,len(normalized)):
    image= sitk.GetImageFromArray(normalized[i])
    image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.swapaxes(image_arr,0,2)
    image=sitk.GetImageFromArray(image_arr)
    #sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/norm_imgs/norm'+imgs[i]+'.hdr')
    sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/norm_imgs/norm'+imgs[i]+'.hdr')


