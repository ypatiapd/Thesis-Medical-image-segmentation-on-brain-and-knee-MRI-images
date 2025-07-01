# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:46:39 2022

@author: ypatia
"""

import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math

Y, X = svm_read_problem('dna.tr.txt')
y_t, x_t = svm_read_problem('dna.t.txt')
l_train =len(X)
l_test= len(y_t)
#s=np.size(X,1)

start_time=int(round(time.time() * 1000))
K=3
nr_feature = 0
dict_keys = list()

total_pred=list() #predictions for j instances and k classes , j*k size 

for i in range(1,K+1): # 1 to all class method 
    y=Y.copy()
    
    for j in range(0,l_train): #keep the i class as 1 and all the others as -1
        if (Y[j] == i):
            y[j]=1
        else:
            y[j]=-1
             
    y_set=list()  # set of y estimates for each anotator for calculate the mis afterwards
    
    B=50
    for _ in range(0,B):  #so B=10000
    
        sample_y = list()
        sample_x = list()
        sample = np.random.randint(0, l_train, size=1000) # bootstrap sample for the current weak anotator
        for j in range(0,len(sample)):
            sample_x.append(X[sample[j]])
            sample_y.append(y[sample[j]])
        #print(len(sample_x))
        #print(len(sample_y))
            #print(sample_y(i))
        prob = problem(sample_y, sample_x)  #solve simple SVM for each weak anotator
        param = parameter('-s 0 -c 4 -B 1')
        m = train(prob, param)
        save_model('protein.model', m)
        m = load_model('protein.model')
        #w= list()  den ta xreiazomaste auta ta w 
        #for j in range(0,60000):     
        #    w.append(m.get_decfun_coef(feat_idx=j))
        #w_set.append(w)
        #print(sample_x)
        y_t, x_t = svm_read_problem('dna.t.txt')
        y_test=y_t.copy()
        for j in range(0,l_test):
            if (y_t[j] == i):
                y_test[j]=1
            else:
                y_test[j]=-1
                
        #print(len(y_t))
        p_label, p_acc, p_val = predict(y_test, x_t, m, '-b 1')  #predict each model to the same test data
        print(len(p_label))
        ACC, MSE, SCC = evaluations(y_test, p_label)
        y_set.append(p_label);  #save y estimates for later
        
    mi=list()   #weight of each anotator in linear aggregation
    for j in range(0,B): # number of annotators hence number of mixing coeffs
            mi.append(1/B)
    for k in range (0,3):  # k number of iterations , hopefully convergence
        #print (k)
        #print ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        w_tilde_set=list()  # set twn w transformed
        w_fixed_set=list()  # set twn w epithimitwn
        y_t, x_t = svm_read_problem('dna.t.txt') #test data
        
        
        for j in range (0,l_test):
            dict_keys.append(x_t[j].keys())
        #we solve equation 13 page 12 li et al
        #compute a simple svm problem for each weak annotator, keep the w tilde  in a set of vectors
        #then find w_fixed set and then the mis for the current iteration
        for j in range(0,B): 
            y_t, x = svm_read_problem('dna.t.txt') #test data , idio me to x_t
            
            for q in range(0,l_test):
                a=0
                a=np.sqrt(mi[j])*y_set[j][q]*y_set[0][q] 
                #print(a)
                #print('one')
                for z in dict_keys[q]: 
                    x[q][z]=a*x[q][z]  #Xis transformed for j annotator 
                    #print(x[q][z])
           
            prob = problem(y_set[0], x)  #we pas y1 to each model, it gets simplified with the transformed Xis and in the machine we have yj 
            param = parameter('-s 0 -c 4 -B 1')
            m2 = train(prob, param)
            #save_model('protein2.model', m2)
            #m2 = load_model('protein.model')
            w_tilde= list() 
            max_len=0
            for q in range(0,l_test):
                if len(x_t[q])>max_len:
                    max_len= len(x_t[q])
            nr_feature =  m2.get_nr_feature() #number of features, also the length of w vector
            for z in range(0,nr_feature):      #get w tilde coeffs
                w_tilde.append(m2.get_decfun_coef(feat_idx=z)) #tsekare gia class 1 kai 2 an vgazei ta idia w . Tha prepe giati exw binary provlima
            w_tilde_set.append(w_tilde)   #save in w tilde set
            
            #m2.get_labels() auto epistrefei enan pinaka me tis 2 klaseis, kai o idx tou pinaka autou pou periexei tin  klasi pou theloume(1) , einai 0 
            #w_tilde_set.append(m2.get_decfun(label_idx=0))   #auto kalitera, alla epistrefei tuple anti gia list, ftiakstoooo
        for j in range(0,B): # fix the w tilde back to w 
            w_fixed_set.append(np.multiply(w_tilde_set[j],np.sqrt(mi[j]))) # fixing w_tilde , but what comes first , μ or wt
            
        sum_inner_product=0
        for j in range(0,B):
            sum_inner_product=sum_inner_product+np.inner(w_fixed_set[j],w_fixed_set[j])
            
        print('mis')
        for j in range(0,B): #compute the mis of the next iteration with the norms equasion  li et al 
            inner_product=np.inner(w_fixed_set[j],w_fixed_set[j])
            mi[j]=inner_product/sum_inner_product # edw to lathos vgainoun idia ta m
            print(mi[j])
       
    k_pred=list()
    
        
    for z in range(0,l_test):
    #    sum=0
    #    for q in range(0,B):
    #        sum=sum+mi[q]*np.inner(w_fixed_set[q],x_t[j])
        x_sparse = list()
        for j in range (0,nr_feature):
            if (j in dict_keys[z]):
                x_sparse.append(x_t[z][j])
            else:
                x_sparse.append(0)
        sum1=0
        for q in range(0,B):
            sum1=sum1+mi[q]*np.inner(w_fixed_set[q],x_sparse)
        k_pred.append(sum1)
    total_pred.append(k_pred)
    

class_labels = list()
    
for i in range(0,l_test):
    max2 = -600000000
    classl = 0
    for j in range(0,K):
        if max2 < total_pred[j][i]:
            max2 = total_pred[j][i]
            classl = j+1
    class_labels.append(classl)        
        
#class_labels / y_t

sum1 = 0
for i in range(0,len(class_labels)):
    if (class_labels[i] == y_t[i]) :
        sum1 = sum1 + 1

print (sum1/len(class_labels))      
    

#theloume na kanoume to eswteriko ginomeno w_fixed_set me to x_t[j]
#to w_fixed_set exei 60000 as poume mikos , enw to x_t einai se sparse morfi
#opote theloume na dimiourgoume gia kathe j ena x mikous 60000 me midenika, 
#kai opou exei timi to x_t, to vlepoume apo ta keys, tin prosthetoume sto x
   

duration=int(round(time.time() * 1000)) - start_time

print('Time:',duration)

#print(mi)