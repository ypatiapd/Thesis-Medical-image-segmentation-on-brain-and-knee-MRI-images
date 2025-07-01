# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:43:54 2023

@author: ypatia
"""


import SimpleITK as sitk
imgs=['01']#,'04','05','12','20','29','13','39','33','28']
import numpy as np
from PIL import Image

for ite in imgs:  
    
   
    imageName='strip'+ite
   
# Open the image file in read mode
    with open(imageName, 'rb') as f:
        img = Image.open(f)

# Convert the image to a NumPy array
    image_arr = np.array(img)
    print(len(image_arr))

    nan_indices = [i for i, d in enumerate(image_arr) if any(np.isnan(list(d.values())))]
    arr = [d for i, d in enumerate(image_arr) if i not in nan_indices]
    print(len(arr))

    image = sitk.GetImageFromArray(arr)

    #sitk.WriteImage(image, 'C:/Users/ypatia/diplomatiki/denoised/denoised'+ite+'.hdr')