# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:56:50 2022

@author: jaime
"""

#TTD
#min diavazeis oli tin wra ta data
#trekse se polla datasets
#ilopoihsh se libsvm
#pare ton arithmo klasewn me sinartisi

import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math

#RESULTS
#first test , dna dataset,  data size: 2586  ,  acc: ~85%    10 annotators

#comparing to Yan results

#dna dataset,  data size: 3186  ,  acc: ~85%    10 annotators ,  50 iterations for mi convergence  ,acc= 86%



## READ THE DATA 


filename=('rcv1_train.multiclass')

K=53

Y, X = svm_read_problem(filename)

array = [0] * K

for j in range(1,K+1):
    for i in range(0,len(Y)):
        if j == Y[i]:
            array[j-1] = array[j-1]+1
            
classes_out = list()
classes_in = list()
for i in range(0,len(array)):
    if array[i] <= 11:
        classes_out.append(i+1)
    else:
        classes_in.append(i+1)
        

fin = open(filename , 'rt')
fout = open('rcv_train_beta.txt','wt')


for line in fin:
    x = line.split(" ")
    if (int(x[0]) not in classes_out):
       fout.write(line) 
    #else:
        #print(x[0])

fin.close()
fout.close()

fin = open('rcv1_test.multiclass','rt')   
fout = open('rcv_train_gama.txt','wt')


for i in classes_in:
    fin = open('rcv1_test.multiclass','rt')   
    index= 78
    #print(i)
    for line in fin:
        x = line.split(" ")
        #print(int(x[0]))
        if (int(x[0]) == i):
           #print(i)
           fout.write(line) 
           index-=1
        if index == 0: 
            break
    fin.close()
    
fin = open('rcv_train_beta.txt','rt')   

index=14440
for line in fin:
    fout.write(line) 
    index-=1
    if index == 0: 
        break
    
fin.close()
fout.close()


    

            