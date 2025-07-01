# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:08:44 2022

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:14:13 2022

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


fin = open('9090290.txt' , 'rt')
fout = open('xtipiti2.txt','wt')

for line in fin:
    line = line[:-1]

    x = line.split(",")
    # row = list()
    
    row=str(x[0]+' ')
    
    for i in range(1,len(x)):
        if x[i] != '0' and x[i] != '0 ':
            row = row + str(i)+":"+x[i]+' '
            # row.append(str(i)+":"+x[i])
        
    # row = str(row)
    fout.write(row) 
    fout.write('\n')      
     
    #print(x)
    #print(line)

"""with open('data_csv.csv', 'a+', newline='') as file:
    writer = csv.writer(file)
    for i in range()       


    row= list()
    
    writer.writerow(row)"""