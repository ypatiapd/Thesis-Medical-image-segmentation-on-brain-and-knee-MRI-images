# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:44:48 2022

@author: jaime
"""

from numpy import array
from scipy.sparse import csr_matrix
import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math
import csv


filename=('connect-4.txt')

fin = open(filename , 'rt')

list1 = list()
sample_size = 10000

for line in fin:
    list1.append(line)

sample = np.random.choice(len(list1), size=sample_size, replace=False)
#sample = sample.sort()
# list2 = list1[sample]

list2 = list()


for i in range(0,len(sample)):
    list2.append(list1[sample[i]])
          
print(len(list2))

fout = open('connect-sample3-10k.txt','wt')

for line in list2:
    fout.write(line) 

fin.close()
fout.close()