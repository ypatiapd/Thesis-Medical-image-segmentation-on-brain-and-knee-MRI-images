# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:11:08 2023

@author: ypatia
"""


"""
Created on Sun Jul 17 20:18:05 2022

@author: jaime
"""
import logging
import os
import SimpleITK as sitk
import six
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import time 
import numpy as np
import six
import dipy 
import warnings
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score,f1_score
import liblinear as liblinear
from liblinear.liblinearutil import *

# Create a logger object
logger = logging.getLogger(__name__)

# Create a file handler that writes to a file called 'output.log'
file_handler = logging.FileHandler('liblinear_B4_mae_24featnew.log')

# Create a stream handler that writes to the console 
stream_handler = logging.StreamHandler()

# Add the handlers to the logger object
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Set the logging level to INFO
logger.setLevel(logging.INFO)
#from libsvm.svmutil import *
## READ THE DATA 

#19,31,40,42 eksw

K=3
iterations=1

classes=[1, 2, 3]  

#class2: 13,33,28
#class1: 25,26,29
#class3: 13,23,26

np.random.seed(42)

filenames=list()

mean_all_dice=0
mean_accuracy=0
mean_f1=0
mean_recall=0
mean_precision=0
mean_recall1=0
mean_recall2=0
mean_recall3=0
mean_precision1=0
mean_precision2=0
mean_precision3=0

#annoteitors me diaforetika features!!!!!!!!!!!!!!

filename01=['strip13','strip39','strip28','strip33']#,'strip16']#,'strip22']#,'strip16','strip22']
filenames.append(filename01)

filename02=['strip22','strip38','strip09','strip04']#,'strip33']#,'strip25']#,'strip25','strip07']
filenames.append(filename02)

filename04=['strip09','strip20','strip38','strip14']#,'strip22']#,'strip17']#,'strip05','strip25']#,'strip33','strip26'] # 90
filenames.append(filename04)

filename05=['strip38','strip07','strip25','strip14']#,'strip22']#,'strip04']#,'strip04','strip09'] # 90
filenames.append(filename05)

filename06=['strip38','strip25','strip07','strip09']#,'strip05']#,'strip37']#,'strip05','strip37'] # 90
filenames.append(filename06)

filename07=['strip38','strip14','strip05','strip34']#,'strip25']#,'strip09']#,'strip09','strip37'] # 90
filenames.append(filename07)

filename09=['strip25','strip04','strip38','strip14']#,'strip37']#,'strip07']#,'strip17'] # 90
filenames.append(filename09)

filename12=['strip04','strip20','strip05','strip22']#,'strip17']#,'strip37']#,'strip37'] # 90
filenames.append(filename12)

filename13=['strip01','strip39','strip28','strip33']#,'strip22']#,'strip16']#,'strip04'] # 90
filenames.append(filename13)

filename14=['strip34','strip37','strip17','strip25']#,'strip38']#,'strip07']#,'strip29'] # 90
filenames.append(filename14)

filename16=['strip32','strip04','strip22','strip20']#,'strip30']#,'strip17']#,'strip33'] # 90
filenames.append(filename16)

filename17=['strip14','strip37','strip09','strip04']#,'strip22']#,'strip20']#,'strip20'] # 90
filenames.append(filename17)

filename20=['strip04','strip22','strip38','strip09']#,'strip33']#,'strip17']#,'strip29'] # 90
filenames.append(filename20)

filename21=['strip16','strip14','strip39','strip13']#,'strip07']#,'strip34']#,'strip34'] # 90
filenames.append(filename21)

filename22=['strip33','strip04','strip05','strip20']#,'strip38']#,'strip17']#,'strip17']#,'strip26','strip07']
filenames.append(filename22)

#filename23=['strip33','strip22','strip07']#,'strip04','strip01','strip39']#,'strip39'] # 90
#filenames.append(filename23)

filename25=['strip38','strip37','strip14','strip09']#,'strip07']#,'strip05']#,'strip05'] # 90
filenames.append(filename25)

#filename26=['strip38','strip25','strip09']#,'strip04','strip22','strip33']#,'strip29'] # 90
#filenames.append(filename26)

filename27=['strip38','strip07','strip34','strip25']#,'strip33']#,'strip09']#,'strip09'] # 90
filenames.append(filename27)

filename28=['strip01','strip33','strip13','strip39']#,'strip22']#,'strip04']#,'strip04'] # 90
filenames.append(filename28)

#filename29=['strip25','strip38','strip37']#,'strip14','strip07','strip05']#,'strip34'] # 90
#filenames.append(filename29)

#filename30=['strip29','strip37','strip14']#,'strip04','strip34','strip05']#,'strip07'] # na vgei
#filenames.append(filename30)

#filename31=['strip21','strip16','strip30','strip37','strip09','strip14']#,'strip34'] # na vgei
#filenames.append(filename31)

#filename32=['strip16','strip23','strip33']#,'strip22','strip04','strip01']#,'strip13'] # 90
#filenames.append(filename32)

filename33=['strip22','strip38','strip04','strip20']#,'strip07']#,'strip28']#,'strip07'] # 90
filenames.append(filename33)

filename34=['strip14','strip38','strip07','strip25']#,'strip37']#,'strip09']#,'strip09'] # 90
filenames.append(filename34)

filename37=['strip25','strip14','strip17','strip34']#,'strip09']#,'strip38']#,'strip38'] # 90
filenames.append(filename37)

filename38=['strip25','strip07','strip34','strip14']#,'strip09']#,'strip05']#,'strip26'] # 90
filenames.append(filename38)

filename39=['strip13','strip01','strip33','strip28']#,'strip04']#,'strip23']#,'strip26'] # 90
filenames.append(filename39)




target_filenames=list()

target_filenames.append('strip01')
target_filenames.append('strip02')
target_filenames.append('strip04')
target_filenames.append('strip05')
target_filenames.append('strip06')
target_filenames.append('strip07')
target_filenames.append('strip09')
target_filenames.append('strip12')

target_filenames.append('strip13')

target_filenames.append('strip14')
target_filenames.append('strip16')
target_filenames.append('strip17')

target_filenames.append('strip20')

target_filenames.append('strip21')
target_filenames.append('strip22')
#target_filenames.append('strip23')
target_filenames.append('strip25')
#target_filenames.append('strip26')
target_filenames.append('strip27')
target_filenames.append('strip28')

#target_filenames.append('strip29')

#target_filenames.append('strip30')

#target_filenames.append('strip31')
#target_filenames.append('strip32')
target_filenames.append('strip33')
target_filenames.append('strip34')
target_filenames.append('strip37')
target_filenames.append('strip38')
target_filenames.append('strip39')


B=len(filenames[0])

#dokimase diaforetiko arithmo annotators gia kathe eikona 
     


for item in range(0,len(target_filenames)):

    ulabeled_x = list()
    ulabeled_y = list()  
    
    test_x =list()
    test_y = list()
    
    Y2 = list()
    X2 = list()
    
    #Y2, X2 = svm_read_problem(target_filenames[item])
    Y20, X20 = svm_read_problem(target_filenames[item])
    nan_indices = [i for i, d in enumerate(X20) if any(np.isnan(list(d.values())))]
    X2 = [d for i, d in enumerate(X20) if i not in nan_indices]
    Y2 = [label for i, label in enumerate(Y20) if i not in nan_indices]
    
    del Y20,X20,nan_indices
        
    ltrain_size=10000
    utrain_size = 100000
    test_size = len(Y2)
    
    ss = np.random.choice(len(Y2), size=len(Y2), replace=False)  #pairnoume megalutero dianysma apo oso xreiazomaste gia na apofugoume ta error logw deikth ite
    
    for i in range(0,utrain_size):
            
        temp=dict()
        temp[1]=X2[ss[i]][1] #brightness
        temp[2]=X2[ss[i]][2] #lbp
        #temp[3]=X2[ss[i]][3] #kyrtosis
        #temp[4]=X2[ss[i]][4] #mean hist
        #temp[5]=X2[ss[i]][5] #gradient magn
        #temp[6]=X2[ss[i]][6] #gradient orientation
        temp[7]=X2[ss[i]][7] #x 
        temp[8]=X2[ss[i]][8] #y
        temp[9]=X2[ss[i]][9] #z
        temp[10]=X2[ss[i]][10] #euclidean
        
        temp[11]=X2[ss[i]][11]# autocorrelation
        temp[12]=X2[ss[i]][12]# idn
        
        temp[13]=X2[ss[i]][13]# highGrayLevel
        temp[14]=X2[ss[i]][14]# LowGrayLevel
        temp[15]=X2[ss[i]][15]# LongRun
        temp[16]=X2[ss[i]][16]# shortRun
        
        temp[17]=X2[ss[i]][17] #highGrayLevel
        temp[18]=X2[ss[i]][18] #LowGrayLevel
        temp[19]=X2[ss[i]][19] #LargeDependence
        
        temp[20]=X2[ss[i]][20] #10perc
        temp[21]=X2[ss[i]][21] #90perc
        temp[22]=X2[ss[i]][22] #Energy
        temp[23]=X2[ss[i]][23] #Mean
        temp[24]=X2[ss[i]][24] #Max
        temp[25]=X2[ss[i]][25] #Min
        temp[26]=X2[ss[i]][26] #Median
        temp[27]=X2[ss[i]][27] #RootMeanSquare
        temp[28]=X2[ss[i]][28] #TotalEnergy
        
        
        ulabeled_x.append(temp)
        ulabeled_y.append(Y2[ss[i]])
            #ulabeld_y.append(copy.copy())
    
        
    for i in range(0,test_size):
        
        temp=dict()
        temp[1]=X2[i][1] #brightness
        temp[2]=X2[i][2] #lbp
        #temp[3]=X2[i][3] #kyrtosis
        #temp[4]=X2[i][4] #mean hist
        #temp[5]=X2[i][5] #gradient magn
        #temp[6]=X2[i][6] #gradient orientation
        temp[7]=X2[i][7] #x 
        temp[8]=X2[i][8] #y
        temp[9]=X2[i][9] #z
        temp[10]=X2[i][10] #euclidean
        
        temp[11]=X2[i][11]# autocorrelation
        temp[12]=X2[i][12]# idn
        
        temp[13]=X2[i][13]# highGrayLevel
        temp[14]=X2[i][14]# LowGrayLevel
        temp[15]=X2[i][15]# LongRun
        temp[16]=X2[i][16]# shortRun
        
        temp[17]=X2[i][17] #highGrayLevel
        temp[18]=X2[i][18] #LowGrayLevel
        temp[19]=X2[i][19] #LargeDependence
        
        temp[20]=X2[i][20] #10perc
        temp[21]=X2[i][21] #90perc
        temp[22]=X2[i][22] #Energy
        temp[23]=X2[i][23] #Mean
        temp[24]=X2[i][24] #Max
        temp[25]=X2[i][25] #Min
        temp[26]=X2[i][26] #Median
        temp[27]=X2[i][27] #RootMeanSquare
        temp[28]=X2[i][28] #TotalEnergy
        
        
        
        test_x.append(temp)
        test_y.append(Y2[i])
        #test_x.append(copy.copy(X2[ss[i]]))
        #test_y.append(copy.copy(Y2[ss[i]]))
    
    print('starting')   
    start_time=int(round(time.time() * 1000))
    
    nr_feature = 0 #number of features
    dict_keys = list()
    
    total_pred=list() #predictions for j instances and k classes , j*k size 
    
    ##RUN K TIMES FOR ALL CLASSES
    #tr_size = ltrain_size/len(filenames) 
    
     # bootstrap sample for the current weak anotator
    samples=list()
    lab_x_all=list()
    lab_y_all=list()
    
    
    for i in range(0,B):
        print('reading file ')
        Y10, X10 = svm_read_problem(filenames[item][i])
        print('end read')
        nan_indices = [i for i, d in enumerate(X10) if any(np.isnan(list(d.values())))]
        X1 = [d for i, d in enumerate(X10) if i not in nan_indices]
        Y1 = [label for i, label in enumerate(Y10) if i not in nan_indices]
        print('end nan')
        del Y10,X10,nan_indices
        #Y1, X1 = svm_read_problem(filenames[item][i])
        ss = np.random.choice(len(Y1), size=len(Y1), replace=False)  
        
        counter1= 0
        counter2= 0
        counter3= 0
        
        ite = 0
        
        w1=10
        w2=10
        w3=10
        labeled_x = list()
        labeled_y = list()
        
        while ite < ltrain_size:
            labeled_y.append(copy.copy(Y1[ss[ite]]))
            labeled_x.append(copy.copy(X1[ss[ite]]))
            ite = ite + 1
            
        '''while counter1+counter2+counter3 < ltrain_size:
             if Y1[ss[ite]] == 1 and counter1 < w1*ltrain_size/30:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter1 = counter1 + 1
             if Y1[ss[ite]] == 2 and counter2 <w2*ltrain_size/30:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter2 = counter2 + 1
             if Y1[ss[ite]] == 3 and counter3 < w3*ltrain_size/30:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter3 = counter3 + 1  
             ite = ite + 1'''
        lab_x_all.append(labeled_x)
        lab_y_all.append(labeled_y)
        
    for i in classes:
                 
        y_set=list()  # set of y estimates for each anotator for calculate the mis afterwards, size=B*utrain_size
        
        ##TRAIN B ANNOTATORS AND GET Y SET FOR PROBLEM 8
        
        for h in range(0,B):  #train B weak annotators on labeled data and predict on unlabeled data
            
            labeled_x = lab_x_all[h]
            labeled_y = lab_y_all[h]
            
            y=list()
            y=labeled_y.copy() #k auto perittouli ftiaksto
            
            for j in range(0,ltrain_size): #keep the i class as 1 and all the others as -1
                if (labeled_y[j] == i):
                    y[j]=1
                else:
                    y[j]=-1
            sample_y = list()
            sample_x = list()
            
            for j in range(0,len(labeled_x)):
               
                temp=dict()
                temp[1]=labeled_x[j][1] #Brightness
                temp[2]=labeled_x[j][2] #lbp
                #temp[3]=labeled_x[j][3] #Kyrtosis
                #temp[4]=labeled_x[j][4] #meanHistogram
                #temp[5]=labeled_x[j][5]  #gradient magn       
                #temp[6]=labeled_x[j][6]  #gradient orientation
                temp[7]=labeled_x[j][7]  #x
                temp[8]=labeled_x[j][8]  #y
                temp[9]=labeled_x[j][9]  #z
                temp[10]=labeled_x[j][10] #euclidian
                
                temp[11]=labeled_x[j][11] #autocorrelation
                temp[12]=labeled_x[j][12] #idn
                
                temp[13]=labeled_x[j][13] #highGrayLevel  
                temp[14]=labeled_x[j][14] #LowGrayLevel
                temp[15]=labeled_x[j][15] #LongRun
                temp[16]=labeled_x[j][16] #ShortRun
                
                temp[17]=labeled_x[j][17] #HighgrayLevel
                temp[18]=labeled_x[j][18] #LowgrayLevel
                temp[19]=labeled_x[j][19] #LargeDependence
                
                temp[20]=labeled_x[j][20] #10perc
                temp[21]=labeled_x[j][21] #90perc
                temp[22]=labeled_x[j][22] #energy
                temp[23]=labeled_x[j][23] #mean
                temp[24]=labeled_x[j][24] #max
                temp[25]=labeled_x[j][25] #min
                temp[26]=labeled_x[j][26] #median
                temp[27]=labeled_x[j][27] #rootMeanSquare
                temp[28]=labeled_x[j][28] #totalEnergy
                
                
                sample_x.append(temp)
                sample_y.append(copy.copy(y[j])) 
            
            prob = problem(sample_y, sample_x)  #solve simple SVM for each weak anotator
            
           
            param = parameter('-s 1 -c 1') #s=1 gia dual problem ,-C gia na vrei to kalitero c
            
            
            m = train(prob, param) #train on labeled data
           
            y_test = list() # gia na vroume to acc twra gia tous weak kanonika den tha to exoume
            
            for j in range(0,utrain_size):
                if (ulabeled_y[j] == i):
                    y_test.append(1)
                else:
                    y_test.append(-1)
            p_label, p_acc, p_val = predict(y_test, ulabeled_x, m)  #predict each model to the same test data
            #p_label, p_acc, p_val = svm_predict(y_test, ulabeled_x, m)  #predict each model to the same test data
            
            TP = 0
            FN = 0
            for k in range(0,len(ulabeled_y)):
                if p_label[k] == 1 and y_test[k] == 1:
                    TP += 1
                elif p_label[k] != 1 and y_test[k] == 1:
                    FN += 1
            print(TP)
            print(FN)
            #print("Recall e precision a oxi recall einai auto")
            print('recall = '+format(TP/(TP+FN)))               
            
            
            ACC, MSE, SCC = evaluations(y_test, p_label)
            y_set.append(p_label);  #save y estimates for later
            
        mi=list()   #weight of each anotator in linear aggregation
        for j in range(0,B): # number of annotators hence number of mixing coeffs
                mi.append(1/B)  #warm start
                
        #SOLVE PROBLEM 8 TO FIND OPTIMAL MIS , FOR k ITERATIONS UNTIL CONVERGENCE
        for k in range (0,iterations):  # k number of iterations , hopefully convergence
            
           
            w_tilde_set=list()  # set twn w transformed
            w_fixed_set=list()  # set twn w epithimitwn
            
            dict_keys=list() # kleidia twn x timwn 
            for j in range (0,utrain_size):
                dict_keys.append(ulabeled_x[j].keys())
            #we solve equation 13 page 12 li et al
            #compute a simple svm problem for each weak annotator, keep the w tilde  in a set of vectors
            #then find w_fixed set and then the mis for the current iteration
            for j in range(0,B): 
                x=list()
                for q in range(0,utrain_size):    
                    x.append(copy.copy(ulabeled_x[q]))
                               
                for q in range(0,utrain_size):  #transform x unlabeled data
                    a=0
                    a=np.sqrt(mi[j])*y_set[j][q]*y_set[0][q] 
                    for z in dict_keys[q]: 
                        x[q][z]=a*x[q][z]  #Xis transformed for j annotator 
                        
                prob = problem(y_set[0], x)  #we pass y1 to each model, it gets simplified with the transformed Xis and in the machine we have yj 
                
                param = parameter('-s 2 -c 1')  # na mhn pesei katw apo 10 to -c, palia 100
    
                m2 = train(prob, param)
                #save_model('protein2.model', m2)
                w_tilde= list() 
               
                nr_feature =  m2.get_nr_feature() #number of features, also the length of w vector
                for z in range(0,nr_feature):      #get w tilde coeffs
                    w_tilde.append(m2.get_decfun_coef(feat_idx=z)) #tsekare gia class 1 kai 2 an vgazei ta idia w . Tha prepe giati exw binary provlima
                w_tilde_set.append(w_tilde) #save in w tilde set
                
            for j in range(0,B): # fix the w tilde back to w 
                w_fixed_set.append(np.multiply(w_tilde_set[j],np.sqrt(mi[j]))) # fixing w_tilde , but what comes first , μ or wt
                
            sum_inner_product=0
            for j in range(0,B):
                sum_inner_product=sum_inner_product+np.inner(w_fixed_set[j],w_fixed_set[j])
                
            for j in range(0,B): #compute the mis of the next iteration with the norms equasion  li et al 
                inner_product=np.inner(w_fixed_set[j],w_fixed_set[j])
                mi[j]=inner_product/sum_inner_product # calculate mis for the current iteration
            print(mi)
           
        #CALCULATE PREDICTIONS FOR EACH CLASS
        k_pred=list()  #predictions array for the test data for the current iteration ,size= 1*test_size
            
        dict_keys = list() #keys twn test x data
        
        for j in range (0,test_size):
            dict_keys.append(test_x[j].keys())
            
        for z in range(0,test_size): #metatrepoume to x se sparse gia na borei na ginei eswteriko ginomeno me to w ,pou periexei times gia sxedon ola ta features
            x_sparse = list()
            for j in range (0,nr_feature):
                if (j in dict_keys[z]):
                    x_sparse.append(copy.copy(test_x[z][j]))
                else:
                    x_sparse.append(0)
            sum1=0
            for q in range(0,B):
                sum1=sum1+mi[q]*np.inner(w_fixed_set[q],x_sparse)
            k_pred.append(sum1) #provlepsi gia to z instance tou test dataset
        total_pred.append(k_pred) #prosthiki ston sinoliko pinaka twn predictions , mia seira tou pinaka diladi apo tis K seires sinolika
        
    class_labels = list()  #pinakas me labels gia ola ta instances, size=1*test_size
        
    for i in range(0,test_size): #calculate the biggest prediction for each instance
        max2 = -600000000
        classl = 0
        for j in range(0,K):
            if max2 < total_pred[j][i]:
                max2 = total_pred[j][i]
                classl = classes[j]
        class_labels.append(classl)  #the class that has the biggest prediction is the class label for the instance 
    
    sum1 = 0
    sumlab1 = 0
    sumlab2 = 0
    sumlab3 = 0
    TP1 = 0
    TP2 = 0
    TP3 = 0
    
    for i in range(0,len(class_labels)):
        if (class_labels[i] == test_y[i]) :
            sum1 = sum1 + 1
            if test_y[i]==1:
                sumlab1 +=1
                TP1 += 1 
            elif test_y[i]==2:
                sumlab2 +=1
                TP2 += 1 
            elif test_y[i]==3:
                sumlab3 +=1
                TP3 += 1
    
    final_acc= sum1/len(class_labels) #accuracy of model
    print ('Total Accuracy: '+format(sum1/len(class_labels)))  
    
    print('Class1 precision: ' +format(sumlab1/class_labels.count(1)))#posa 1 perissevoun
    print('Class2 precision: '+format( sumlab2/class_labels.count(2)))
    print('Class3 precision: '+format(sumlab3/class_labels.count(3)))
    
    print('Class1 recall: ' +format(sumlab1/test_y.count(1)))#posa 1 leipoun 
    print('Class2 recall: '+format( sumlab2/test_y.count(2)))
    print('Class3 recall: '+format(sumlab3/test_y.count(3)))
    
    print('test1: '+format(test_y.count(1)) +' test2: '+format(test_y.count(2))+ ' test3: '+format(test_y.count(3)))
    print('lab1: '+format(labeled_y.count(1)) +' lab2: '+format(labeled_y.count(2))+ ' lab3: '+format(labeled_y.count(3)))
    print(' ulab1: '+format(ulabeled_y.count(1)) +' ulab2: '+format(ulabeled_y.count(2))+ ' ulab3: '+format(ulabeled_y.count(3)))
    print(' class1_labels: '+format(class_labels.count(1)) +' class2_labels: '+format(class_labels.count(2)) + ' class3_labels: '+format(class_labels.count(3)))
    
    f1=f1_score(test_y, class_labels, average='weighted')
    total_precision=precision_score(test_y,class_labels,average='weighted')
    total_recall=recall_score(test_y,class_labels,average='weighted')
    precision1=sumlab1/class_labels.count(1)
    precision2=sumlab2/class_labels.count(2)
    precision3=sumlab3/class_labels.count(3)
    recall1=sumlab1/test_y.count(1)
    recall2=sumlab2/test_y.count(2)
    recall3=sumlab3/test_y.count(3)
    
    # Load the true and predicted labels as 1D lists
    true_labels = np.array(test_y)
    predicted_labels = np.array(class_labels)

    # Calculate the confusion matrix for each class
    num_classes = 3
    cm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i,j] = np.sum((true_labels == i+1) & (predicted_labels == j+1))

    # Calculate the Dice coefficient for each class
    dice_coeffs = np.zeros((num_classes,))
    for i in range(num_classes):
        tp = cm[i,i]
        fp = np.sum(cm[:,i]) - tp
        fn = np.sum(cm[i,:]) - tp
        dice_coeffs[i] = 2*tp / (2*tp + fp + fn)

   
    mean_dice_coeff = np.mean(dice_coeffs)
    
    print('dice coeff'+format(mean_dice_coeff))
    # Calculate the mean Dice coefficient across all classes
    
    
    mean_all_dice=mean_all_dice+mean_dice_coeff
    mean_accuracy= mean_accuracy+final_acc
    mean_f1=mean_f1+f1
    mean_precision=mean_precision+total_precision
    mean_recall=mean_recall+total_recall
    mean_recall1=mean_recall1+recall1
    mean_recall2=mean_recall2+recall2
    mean_recall3=mean_recall3+recall3
    mean_precision1=mean_precision1+precision1
    mean_precision2=mean_precision1+precision2
    mean_precision3=mean_precision1+precision3
    
    logger.info('Target image: '+format(target_filenames[item]))
    logger.info('Total Accuracy: '+format(sum1/len(class_labels)))
    logger.info(' f1 :'+format( f1_score(test_y, class_labels, average='weighted'))) # For multiclass classification
    logger.info('Total Precision :'+format(precision_score(test_y,class_labels,average='weighted'))) # For multiclass classification
    logger.info('Total Recall :'+format(recall_score(test_y,class_labels,average='weighted'))) # For multiclass classification
    logger.info('Class1 precision: ' +format(sumlab1/class_labels.count(1)))
    logger.info('Class2 precision: '+format( sumlab2/class_labels.count(2)))
    logger.info('Class3 precision: '+format(sumlab3/class_labels.count(3)))
    logger.info('Class1 recall: ' +format(sumlab1/test_y.count(1)))
    logger.info('Class2 recall: '+format( sumlab2/test_y.count(2)))
    logger.info('Class3 recall: '+format(sumlab3/test_y.count(3)))
    logger.info('dice : '+format(mean_dice_coeff))

    logger.info('  ')
    #theloume na kanoume to eswteriko ginomeno w_fixed_set me to x_t[j]
    #to w_fixed_set exei 60000 as poume mikos , enw to x_t einai se sparse morfi
    #opote theloume na dimiourgoume gia kathe j ena x mikous 60000 me midenika, 
    #kai opou exei timi to x_t, to vlepoume apo ta keys, tin prosthetoume sto x
       
    duration=int(round(time.time() * 1000)) - start_time
    
    print('Time:',duration)
    
    
    # -*- coding: utf-8 -*-
    """
    Created on Thu Mar 16 15:01:32 2023

    @author: ypatia
    """

    selected_indices = [i for i, fv in enumerate(test_x) if fv.get(7) == 0.5]
    selected_feature_vectors = [fv for fv in test_x if fv.get(7) == 0.5]
    selected_labels = [test_y[i] for i in selected_indices]

    # Get x and y coordinates from feature vectors
    x_coords = [fv.get(9) for fv in selected_feature_vectors]
    y_coords = [fv.get(8) for fv in selected_feature_vectors]

    # Create scatter plot for ground truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(x_coords, y_coords, c=selected_labels)
    plt.axis('off')
    # Create scatter plot for predicted labels
    plt.subplot(1, 2, 2)
    plt.title("Predicted")
    plt.axis('off')
    selected_class_labels = [class_labels[i] for i in selected_indices]
    plt.scatter(x_coords, y_coords, c=selected_class_labels)

    fig = plt.gcf()
    fig.set_size_inches(10, 5)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    # Show the figure
    plt.show()
    
    
    selected_indices = [i for i, fv in enumerate(test_x) if fv.get(9) == 0.5]
    selected_feature_vectors = [fv for fv in test_x if fv.get(9) == 0.5]
    selected_labels = [test_y[i] for i in selected_indices]

    # Get x and y coordinates from feature vectors
    x_coords = [fv.get(7) for fv in selected_feature_vectors]
    y_coords = [fv.get(8) for fv in selected_feature_vectors]

    # Create scatter plot for ground truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.scatter(x_coords, y_coords, c=selected_labels)
    plt.axis('off')
    # Create scatter plot for predicted labels
    plt.subplot(1, 2, 2)
    plt.title("Predicted")
    plt.axis('off')
    selected_class_labels = [class_labels[i] for i in selected_indices]
    plt.scatter(x_coords, y_coords, c=selected_class_labels)

    fig = plt.gcf()
    fig.set_size_inches(10, 4)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    # Show the figure
    plt.show()
    
    # Select only instances where z=100
    selected_indices = [i for i, fv in enumerate(test_x) if fv.get(9) == 0.5]
    selected_feature_vectors = [fv for fv in test_x if fv.get(9) == 0.5]

    selected_labels = [test_y[i] for i in selected_indices]
    # Get x and y coordinates from feature vectors
    x_coords = [fv.get(7) for fv in selected_feature_vectors]
    y_coords = [fv.get(8) for fv in selected_feature_vectors]

    # Create scatter plot
    plt.scatter(x_coords, y_coords, c=selected_labels)
    plt.show()

    selected_class_labels = [class_labels[i] for i in selected_indices]
    
    plt.scatter(x_coords, y_coords, c=selected_class_labels)
    plt.show()
    

    del class_labels,total_pred,k_pred,x_sparse,dict_keys,w_tilde_set,mi,w_fixed_set,w_tilde,m2,m
    del prob, y_set,x,y_test,labeled_y,labeled_x, lab_x_all,lab_y_all,ulabeled_x,ulabeled_y,test_x,test_y
    del sample_x,sample_y,samples,X1,Y1
    #del selected_indices,selected_class_labels,selected_feature_vectors,x_coords,y_coords

mean_all_dice=mean_all_dice/len(target_filenames)
mean_accuracy=mean_accuracy/len(target_filenames)
mean_f1=mean_f1/len(target_filenames)
mean_precision=mean_precision/len(target_filenames)
mean_recall=mean_recall/len(target_filenames)
mean_recall1=mean_recall1/len(target_filenames)
mean_recall2=mean_recall2/len(target_filenames)
mean_recall3=mean_recall3/len(target_filenames)
mean_precision1=mean_precision1/len(target_filenames)
mean_precision2=mean_precision2/len(target_filenames)
mean_precision3=mean_precision3/len(target_filenames)

logger.info('final mean dice coeff:'+format(mean_all_dice))
logger.info('final mean accuracy:'+format(mean_accuracy))
logger.info('final mean f1 score:'+format(mean_f1))
logger.info('final mean precision:'+format(mean_precision))
logger.info('final mean recall:'+format(mean_recall))
logger.info('final mean recall 1:'+format(mean_recall1))
logger.info('final mean recall 2:'+format(mean_recall2))
logger.info('final mean recall 3:'+format(mean_recall3))
logger.info('final mean precision 1:'+format(mean_precision1))
logger.info('final mean precision 2:'+format(mean_precision2))
logger.info('final mean precision 3:'+format(mean_precision3))

