# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:27:29 2022

@author: ypatia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import itk
import nibabel
import radiomics

#path='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr'
path='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'

#testpath='C:/Users/ypatia/AppData/Local/Temp/pyradiomics/data/brain1_image.nrrd'
#test2path='C:/Users/ypatia/AppData/Local/Temp/pyradiomics/data/brain1_label.nrrd'
# The following line only needs to run once for a user
# to download the necessary binaries to read HDR.

image = nibabel.load(path)
data= image.get_fdata()

sum1=0
sum2=0
data2=list()
list1=list()

#values
'''for i in range(0,256):
    list2=list()
    for j in range(0,256):
        list3=list()
        for k in range(0,160):
            if data[i][j][k][0]!=0:
                print(format(data[i][j][k][0])+' i='+ format(i) + ' j='+ format(j)+' k='+ format(k) )
                sum1+=1
            list3.append(data[i][j][k][0])
        list2.append(list3)
    list1.append(list2)    '''
 
#classes
for i in range(0,176):
    list2=list()
    for j in range(0,208):
        list3=list()
        for k in range(0,176):
            if data[i][j][k][0]!=0:
                #print(format(data[i][j][k][0])+' i='+ format(i) + ' j='+ format(j)+' k='+ format(k) )
                sum1+=1
            list3.append(data[i][j][k][0])
        list2.append(list3)
    list1.append(list2)            
        
#elif data[i][j][k][0]==0:
         

#plot data

#keli i o aksonas x
#keli j o aksonas y
#keli z o aksonas z

x1=list()
y1=list()
x2=list()
y2=list()
x3=list()
y3=list()

#edw plotaroume katopsi

for i in range(0,176):
    for j in range(0,208):   
        if data[i][j][12][0]==1:
            x1.append(i)
            y1.append(j)
        elif data[i][j][12][0]==2:
            x2.append(i)
            y2.append(j)
        elif data[i][j][12][0]==3:
            x3.append(i)
            y3.append(j)
    
arrayx1=np.asarray(x1)
arrayy1=np.asarray(y1)
arrayx2=np.asarray(x2)
arrayy2=np.asarray(y2)
arrayx3=np.asarray(x3)
arrayy3=np.asarray(y3)

plt.scatter(arrayx1, arrayy1,color='red')
plt.scatter(arrayx2, arrayy2,color='blue')
plt.scatter(arrayx3, arrayy3,color='green')
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
plt.show()


#edw plotaroume prosopsi     

"""for i in range(0,176):
    for j in range(0,208):   
        if data[88][j][i][0]==1:
            x1.append(j)
            y1.append(i)
        elif data[88][j][i][0]==2:
            x2.append(j)
            y2.append(i)
        elif data[88][j][i][0]==3:
            x3.append(j)
            y3.append(i)

arrayx1=np.asarray(x1)
arrayy1=np.asarray(y1)
arrayx2=np.asarray(x2)
arrayy2=np.asarray(y2)
arrayx3=np.asarray(x3)
arrayy3=np.asarray(y3)


#plt.xticks(rotation=90)
#plt.yticks(rotation=90)
plt.scatter(arrayx1, arrayy1,color='red')

plt.scatter(arrayx2, arrayy2,color='blue')
plt.scatter(arrayx3, arrayy3,color='green')
plt.tick_params(axis='x',rotation=90)
plt.show()"""

"""imageio.plugins.freeimage.download()
#img = imageio.imread(path, format='ITK')
#img = np.array(img)"""

"""from OpenImageIO import ImageBuf

img = ImageBuf(hdr_path)

# OpenImageIO has a lot of handy functions for manipulating
# and writing the image back out.
# You could also get a numpy array for the pixel data with:
img.get_pixels()"""


"""import cv2

hdr_path='C:/Users/ypatia/diplomatiki/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-2_anon.hdr'
# IMREAD_ANYDEPTH is needed because even though the data is stored in 8-bit channels
# when it's read into memory it's represented at a higher bit depth
img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)"""