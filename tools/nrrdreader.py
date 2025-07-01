# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:48:18 2022

@author: touloski
"""

import numpy as np
import nrrd
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape


# Some sample numpy data
imageName, filename = getTestCase('brain1')

# Write to a NRRD file
#nrrd.write(filename, data)

# Read the data back from file
readdata, header = nrrd.read(filename)
print(readdata.shape)
print(header)


