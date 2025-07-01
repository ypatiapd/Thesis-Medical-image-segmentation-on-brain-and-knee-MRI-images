
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from intensity_normalization.normalize.nyul import NyulNormalize



imgs=['9001104']#,'02','03','04','05','06','07','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','25','26','27','28','29','30','31','32','33','34','35','37','38','39','40','41','42']

image_paths = list()
for z in imgs:
    

    imageName='C:/Users/jaime/Desktop/DenoiKnees/denoi9001104.hdr'
    #imageName='C:/Users/ypatia/diplomatiki/disc1/OAS1_00'+z+'_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_00'+z+'_MR1_mpr_n'+n[counter]+'_anon_sbj_111.hdr'
    image_paths.append(imageName)
images = [nib.load(image_path).get_fdata() for image_path in image_paths]

# normalize the images and save the standard histogram

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
    sitk.WriteImage(image, 'C:/Users/jaime/Desktop/NormiKnees/normi'+imgs[i]+'.hdr')


