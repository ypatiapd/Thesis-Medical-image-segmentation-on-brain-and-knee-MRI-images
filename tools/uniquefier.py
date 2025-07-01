# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:34:11 2022

@author: jaime
"""

import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math




def unique(list1):
 
    # initialize a null list
    unique_list = list()
    
    
    
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    
    
    return (unique_list)


filename=('connect-4.txt')

fin = open(filename , 'rt')


list1 = list()



for line in fin:
    list1.append(line)

list2 = unique(list1)

print(len(list2))

fout = open('connect-unique1.txt','wt')


for line in list2:
    fout.write(line) 
    # fout.write('\n')  

fin.close()
fout.close()

