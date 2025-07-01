
from __future__ import print_function
import numpy as np
import numpy
import SimpleITK as sitk
import six
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

imgs=['9001104']#,'02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']

for z in imgs:    
    #maskName = 'C:/Users/jaime/YanAlgorithm/disc1//OAS1_00'+z+'_MR1/FSL_SEG/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_111_t88_masked_gfc_fseg.hdr'
    
    full_mhd_path = 'C:/Users/jaime/Downloads/MRI_Data/Baseline/KL1/'+z+'/'+z+'.segmentation_masks.mhd'
    
    mask = sitk.ReadImage(full_mhd_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr = np.flip(mask_arr,0)  
    mask = sitk.GetImageFromArray(mask_arr)
    #imageName = 'C:/Users/jaime/Desktop/RegiBets/regi'+z+'.hdr'
    
    imageName = 'C:/Users/jaime/Desktop/KneeImages/knee'+z+'.hdr'
   
    image = sitk.ReadImage(imageName)
    
    data = sitk.GetArrayFromImage(image)
    
    sigma = estimate_sigma(data, N=16)
    
    nlmimage = nlmeans(data, sigma=sigma, patch_radius=1,
                  block_radius=2, rician=False)
    
    image = sitk.GetImageFromArray(nlmimage)
    
    sitk.WriteImage(image, 'C:/Users/jaime/Desktop/DenoiKnees/denoi'+z+'.hdr')
    
