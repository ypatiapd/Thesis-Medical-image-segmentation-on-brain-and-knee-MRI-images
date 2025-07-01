
import SimpleITK
import SimpleITK as sitk
import numpy as np


#imgs=['01','04','05','06','07','09','11','13','16','20','23','25','26','28','29','33','38','39']

imgs=['01','02','04','05','06','07','09','12','13','14','16','17','20','21','22','23','25','26','27','28','29','30','32','33','34','37','38','39']
#n=['4','4','4','4','4','4','3','4','4','4','4','4','4','3','3','4','4','4','3','4','4','4','4','3','4','4','4','4','4','4','4','3','4','4','3','3','4','4','4']
#xalia eikones  3,10,11,15,18,19,31,35,40,41,42
counter=0
images= list()
dists = list()
#image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm01.hdr')
image1 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm12.hdr')


for ite in imgs:      
    
    dists = list()
    # Read in the second image
    #image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')
    image2 = sitk.ReadImage('C:/Users/ypatia/diplomatiki/norm_imgs/norm'+ite+'.hdr')

    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)
    
    
    # Compute the Euclidean distance between the two images
    euclidean_distance = np.sqrt(np.sum((image1_array - image2_array)**2))
    dists.append(euclidean_distance)
    dists.append(ite)
    images.append(dists)
    
    
    
    print("Euclidean distance:", euclidean_distance)

sorted_list = list()
sorted_list = sorted(images, key=lambda x: x[0])
