# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:59:31 2023

@author: ypatia
"""


"""
Created on Sun Jul 17 20:18:05 2022

@author: jaime
""" 
from sklearn.metrics import precision_score, recall_score,f1_score
import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math
import copy 

#from libsvm.svmutil import *
## READ THE DATA 
import logging

# Create a logger object
logger = logging.getLogger(__name__)

# Create a file handler that writes to a file called 'output.log'
file_handler = logging.FileHandler('second_c1_tunning.log')

# Create a stream handler that writes to the console
stream_handler = logging.StreamHandler()

# Add the handlers to the logger object
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

K=4
iterations=1

#1st tunning    
c_values=[0.1,0.2,0.4,0.6,0.8,1,10]

#2nd tunning    
#c_values=[1,2,4,8,10]
tot_precision_2 = 0
tot_precision_4 = 0 
tot_recall_2 = 0
tot_recall_4 = 0

classes=[1, 2, 3,4]  

#class2: 13,33,28
#class1: 25,26,29
#class3: 13,23,26

np.random.seed(42)


filenames=list()

filename=['9036287','9073948','9090290']
filenames.append(filename)


target_filenames=list()
target_filenames.append('9011115')

B=len(filenames[0])

#dokimase diaforetiko arithmo annotators gia kathe eikona 
 


for item in range(0,len(target_filenames)):

    ulabeled_x = list()
    ulabeled_y = list()  
    
    test_x =list()
    test_y = list()
    
    Y2 = list()
    X2 = list()
    
    Y2, X2 = svm_read_problem(target_filenames[item])
    '''Y20, X20 = svm_read_problem(target_filenames[item])
    nan_indices = [i for i, d in enumerate(X20) if any(np.isnan(list(d.values())))]
    X2 = [d for i, d in enumerate(X20) if i not in nan_indices]
    Y2 = [label for i, label in enumerate(Y20) if i not in nan_indices]'''
    
        
    ltrain_size=10000
    utrain_size = 100000
    test_size = 20000
    
    ss = np.random.choice(len(Y2), size=len(Y2), replace=False)  #pairnoume megalutero dianysma apo oso xreiazomaste gia na apofugoume ta error logw deikth ite
    
    for i in range(0,utrain_size):
            
        temp=dict()
        temp[1]=X2[ss[i]][1] #brightness
        temp[2]=X2[ss[i]][2] #eucl1
        temp[3]=X2[ss[i]][3] #eucl2
        #temp[4]=X2[ss[i]][4] #edge
        #temp[5]=X2[ss[i]][5] #lbpa1
        temp[6]=X2[ss[i]][6] #lbpa2
        #temp[7]=X2[ss[i]][7] #lbpb1 
        temp[8]=X2[ss[i]][8] #lbpb2
        temp[9]=X2[ss[i]][9] # i 
        temp[10]=X2[ss[i]][10] # j
        temp[11]=X2[ss[i]][11] # k
        temp[12]=X2[ss[i]][12] # gradient_x
        temp[13]=X2[ss[i]][13] # gradient_y
        temp[14]=X2[ss[i]][14] # gradient_z
        temp[15]=X2[ss[i]][15] # magnitude
        temp[16]=X2[ss[i]][16] # orientation_8
        temp[17]=X2[ss[i]][17] # orientation_18
        
        #temp[18]=X2[ss[i]][18] # 10 percentile
        temp[19]=X2[ss[i]][19] # 90 percentile
        temp[20]=X2[ss[i]][20] # mean
        #temp[21]=X2[ss[i]][21] # maximum
        #temp[22]=X2[ss[i]][22] # median      
        temp[23]=X2[ss[i]][23] # totalenergy
        temp[24]=X2[ss[i]][24] # RootMeanSquared
        temp[25]=X2[ss[i]][25] # minimum
        #temp[26]=X2[ss[i]][26] # range
        
        #temp[27]=X2[ss[i]][27] # autocorrelation
        temp[28]=X2[ss[i]][28] # idmn
        temp[29]=X2[ss[i]][29] # idn
        #temp[30]=X2[ss[i]][30] # idm
        #temp[31]=X2[ss[i]][31] # cluster prominence
        #temp[32]=X2[ss[i]][32] # cluster shade 
        
        #temp[33]=X2[ss[i]][33] # shortrunhighgray
        #temp[34]=X2[ss[i]][34] # highgray
        
        
       
        
        
        
        ulabeled_x.append(temp)
        ulabeled_y.append(Y2[ss[i]])
            #ulabeld_y.append(copy.copy())
    
        
    for i in range(utrain_size,utrain_size+test_size):
        
        temp=dict()
        temp[1]=X2[ss[i]][1] #brightness
        temp[2]=X2[ss[i]][2] #eucl1
        temp[3]=X2[ss[i]][3] #eucl2
        #temp[4]=X2[ss[i]][4] #edge
        #temp[5]=X2[ss[i]][5] #lbpa1
        temp[6]=X2[ss[i]][6] #lbpa2
        #temp[7]=X2[ss[i]][7] #lbpb1 
        temp[8]=X2[ss[i]][8] #lbpb2
        temp[9]=X2[ss[i]][9] # i 
        temp[10]=X2[ss[i]][10] # j
        temp[11]=X2[ss[i]][11] # k
        temp[12]=X2[ss[i]][12] # gradient_x
        temp[13]=X2[ss[i]][13] # gradient_y
        temp[14]=X2[ss[i]][14] # gradient_z
        temp[15]=X2[ss[i]][15] # magnitude
        temp[16]=X2[ss[i]][16] # orientation_8
        temp[17]=X2[ss[i]][17] # orientation_18
        
        #temp[18]=X2[ss[i]][18] # 10 percentile
        temp[19]=X2[ss[i]][19] # 90 percentile
        temp[20]=X2[ss[i]][20] # mean
        #temp[21]=X2[ss[i]][21] # maximum
        #temp[22]=X2[ss[i]][22] # median      
        temp[23]=X2[ss[i]][23] # totalenergy
        temp[24]=X2[ss[i]][24] # RootMeanSquared
        temp[25]=X2[ss[i]][25] # minimum
        #temp[26]=X2[ss[i]][26] # range
        
        #temp[27]=X2[ss[i]][27] # autocorrelation
        temp[28]=X2[ss[i]][28] # idmn
        temp[29]=X2[ss[i]][29] # idn
        #temp[30]=X2[ss[i]][30] # idm
        #temp[31]=X2[ss[i]][31] # cluster prominence
        #temp[32]=X2[ss[i]][32] # cluster shade 
        
        #temp[33]=X2[ss[i]][33] # shortrunhighgray
        #temp[34]=X2[ss[i]][34] # highgray
        
        
        
        test_x.append(temp)
        test_y.append(Y2[ss[i]])
        #test_x.append(copy.copy(X2[ss[i]]))
        #test_y.append(copy.copy(Y2[ss[i]]))
    
    print('starting')   
    start_time=int(round(time.time() * 1000))
    
    nr_feature = 0 #number of features
    
    
    ##RUN K TIMES FOR ALL CLASSES
    #tr_size = ltrain_size/len(filenames) 
    
     # bootstrap sample for the current weak anotator
    samples=list()
    lab_x_all=list()
    lab_y_all=list()
    
    
    for i in range(0,B):
        print('reading file ')
        '''Y10, X10 = svm_read_problem(filenames[item][i])
        print('end read')
        nan_indices = [i for i, d in enumerate(X10) if any(np.isnan(list(d.values())))]
        X1 = [d for i, d in enumerate(X10) if i not in nan_indices]
        Y1 = [label for i, label in enumerate(Y10) if i not in nan_indices]
        print('end nan')'''
       
        Y1, X1 = svm_read_problem(filenames[item][i])
        ss = np.random.choice(len(Y1), size=len(Y1), replace=False)  
        
        counter1= 0
        counter2= 0
        counter3= 0
        counter4= 0
        ite = 0
        
        
        w1 = 10
        w2 = 13
        w3 = 10
        w4 = 7
        
        # acc: 0.70 , [20,5,10,5] # 5 features
        # acc: 0.74 , [25,5,5,5] # 5 features
        
        labeled_x = list()
        labeled_y = list()
        '''if i==3: #9 10 
            w1=5
            w2=5
            w3=10
        elif i==2: #11 8
            w1=5
            w2=10
            w3=5'''
        '''while ite < tr_size:
            labeled_y.append(copy.copy(Y1[ss[ite]]))
            labeled_x.append(copy.copy(X1[ss[ite]]))
            ite = ite + 1'''
        '''   
        for t in range(0,ltrain_size):
            labeled_y.append(copy.copy(Y1[ss[t]]))
            labeled_x.append(copy.copy(X1[ss[t]]))
            
        '''   
        while counter1+counter2+counter3+counter4 < ltrain_size:
             if Y1[ss[ite]] == 1 and counter1 < w1*ltrain_size/40:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter1 = counter1 + 1
             if Y1[ss[ite]] == 2 and counter2 <w2*ltrain_size/40:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter2 = counter2 + 1
             if Y1[ss[ite]] == 3 and counter3 < w3*ltrain_size/40:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter3 = counter3 + 1 
             if Y1[ss[ite]] == 4 and counter4 < w4*ltrain_size/40:
                 labeled_y.append(copy.copy(Y1[ss[ite]]))
                 labeled_x.append(copy.copy(X1[ss[ite]]))
                 counter4 = counter4 + 1 
             ite = ite + 1
        lab_x_all.append(labeled_x)
        lab_y_all.append(labeled_y)
        
    for c_val in c_values:
        
        total_pred=list() #predictions for j instances and k classes , j*k size 
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
                    temp[1]=labeled_x[j][1] # brightness
                    temp[2]=labeled_x[j][2] # eucl1
                    temp[3]=labeled_x[j][3] # eucl2
                    #temp[4]=labeled_x[j][4] # edge
                    #temp[5]=labeled_x[j][5] # lbpa1
                    temp[6]=labeled_x[j][6] # lbpa2                
                    #temp[7]=labeled_x[j][7] # lbpb1
                    temp[8]=labeled_x[j][8] # lbpb2                 
                    temp[9]=labeled_x[j][9] # i
                    temp[10]=labeled_x[j][10] # j
                    temp[11]=labeled_x[j][11]  # k
                    temp[12]=labeled_x[j][12]  # gradient x 
                    temp[13]=labeled_x[j][13]  # gradient y 
                    temp[14]=labeled_x[j][14]  # gradient z
                    temp[15]=labeled_x[j][15]  # magnitude
                    temp[16]=labeled_x[j][16]  # orientation_8
                    temp[17]=labeled_x[j][17]  # orientation_18
                    
                    #temp[18]=labeled_x[j][18]  # 10 percentile
                    temp[19]=labeled_x[j][19]  # 90 percentile 
                    temp[20]=labeled_x[j][20]  # mean
                    #temp[21]=labeled_x[j][21]  # maximum
                    #temp[22]=labeled_x[j][22]  # median
                    temp[23]=labeled_x[j][23]  # totalenergy
                    temp[24]=labeled_x[j][24]  # root
                    temp[25]=labeled_x[j][25]  # minimum 
                    #temp[26]=labeled_x[j][26]  # range
                    
                    #temp[27]=labeled_x[j][27]  # autocorrelation
                    temp[28]=labeled_x[j][28]  # idmn
                    temp[29]=labeled_x[j][29]  # idn
                    #temp[30]=labeled_x[j][30]  # idm
                    #temp[31]=labeled_x[j][31]  # cluster prominence
                    #temp[32]=labeled_x[j][32]  # cluster shade
                    
                    #temp[33]=labeled_x[j][33]  # shortrunhighgray
                    #temp[34]=labeled_x[j][34]  # highray          
                    
                    sample_x.append(temp)
                    sample_y.append(copy.copy(y[j])) 
                
                prob = problem(sample_y, sample_x)  #solve simple SVM for each weak anotator
                '''param = parameter('-s 2 -C ')  # na mhn pesei katw apo 10 to -c, palia 100
                    
                c_params = train(prob, param)
                string= '-s 2 -c '+format(c_params[0])
                print('best_c ='+format(c_params[0]))
                param= parameter(string)'''
               
                param = parameter('-s 1 -c ' +str(c_val)) #s=1 gia dual problem ,-C gia na vrei to kalitero c
                
                #prob = svm_problem(sample_y, sample_x)  #solve simple SVM for each weak anotator
                #param = parameter('-s 1 -c 4 -B 1') #s=1 gia dual problem
                #param = svm_parameter('-t 1 -s 0 -c 1')
                
                '''if i == 2:
                    param = parameter('-s 1 -c 4 -w1 5 -w-1 1')
                else:
                    param = parameter('-s 1 -c 4 -B 0')'''
                m = train(prob, param) #train on labeled data
                #m = svm_train(prob, param) #train on labeled data
                #svm-train -s 3 -p 0.1 -t 0
                #save_model('protein.model', m)
                #m = load_model('protein.model')
               
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
                    '''if i == 3:
                        param = parameter('-s 2 -c 10 -w1 2 ')
                   
                    else:
                        param = parameter('-s 2 -c 10 -B 1')'''
                    '''param = parameter('-s 2 -C ')  # na mhn pesei katw apo 10 to -c, palia 100
                    
                    c_params = train(prob, param)
                    string= '-s 2 -c '+format(c_params[0])
                    print('best_c ='+format(c_params[0]))
                    param= parameter(string)'''
                    param = parameter('-s 2 -c 1 ')  # na mhn pesei katw apo 10 to -c, palia 100
        
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
        sumlab4 = 0
        TP1 = 0
        TP2 = 0
        TP3 = 0
        TP4 = 0
        mis4to2 = 0
        mis2to4 = 0
        
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
                elif test_y[i]==4:
                    sumlab4 +=1
                    TP4 += 1
            if (class_labels[i] == 2 and test_y[i] == 4 ) :
                mis4to2 +=1
            if (class_labels[i] == 4 and test_y[i] == 2 ) :
                mis2to4 +=1    
                    
        
        final_acc= sum1/len(class_labels) #accuracy of model
        print ('Total Accuracy: '+format(sum1/len(class_labels)))  
        
        print('Class1 precision: ' +format(sumlab1/class_labels.count(1)))#posa 1 perissevoun
        print('Class2 precision: '+format( sumlab2/class_labels.count(2)))
        print('Class3 precision: '+format(sumlab3/class_labels.count(3)))
        print('Class4 precision: '+format(sumlab4/class_labels.count(4)))
        
        
        print('Class1 recall: ' +format(sumlab1/test_y.count(1)))#posa 1 leipoun 
        print('Class2 recall: '+format( sumlab2/test_y.count(2)))
        print('Class3 recall: '+format(sumlab3/test_y.count(3)))
        print('Class4 recall: '+format(sumlab4/test_y.count(4)))
        
        
        print('test1: '+format(test_y.count(1)) +' test2: '+format(test_y.count(2))+ ' test3: '+format(test_y.count(3)) +' test4: '+format(test_y.count(4)))
        print('lab1: '+format(labeled_y.count(1)) +' lab2: '+format(labeled_y.count(2))+ ' lab3: '+format(labeled_y.count(3))+' lab4: '+format(labeled_y.count(4)))
        print(' ulab1: '+format(ulabeled_y.count(1)) +' ulab2: '+format(ulabeled_y.count(2))+ ' ulab3: '+format(ulabeled_y.count(3)) +' ulab4: '+format(ulabeled_y.count(4)))
        print(' class1_labels: '+format(class_labels.count(1)) +' class2_labels: '+format(class_labels.count(2)) + ' class3_labels: '+format(class_labels.count(3))+' class4_labels: '+format(class_labels.count(4)))
            
        #theloume na kanoume to eswteriko ginomeno w_fixed_set me to x_t[j]
        #to w_fixed_set exei 60000 as poume mikos , enw to x_t einai se sparse morfi
        #opote theloume na dimiourgoume gia kathe j ena x mikous 60000 me midenika, 
        #kai opou exei timi to x_t, to vlepoume apo ta keys, tin prosthetoume sto x
        print(' Misclassified 4s to class 2: '+format(mis4to2))
        print(' Misclassified 2s to class 4: '+format(mis2to4))
        
        tot_precision_2 = tot_precision_2 + sumlab2/class_labels.count(2)
        tot_precision_4 = tot_precision_4 + sumlab4/class_labels.count(4)
        
        tot_recall_2 = tot_recall_2 + sumlab2/test_y.count(2)
        tot_recall_4 = tot_recall_4 + sumlab4/test_y.count(4)

        
        duration=int(round(time.time() * 1000)) - start_time
        
        print('Time:',duration)
        logger.info('Results for c1 value :'+str(c_val))
        logger.info('Total Accuracy: '+format(sum1/len(class_labels)))
        logger.info(' f1 :'+format( f1_score(test_y, class_labels, average='weighted'))) # For multiclass classification
        logger.info('Total Precision :'+format(precision_score(test_y, class_labels,average='weighted'))) # For multiclass classification
        logger.info('Total Recall :'+format(recall_score(test_y, class_labels,average='weighted'))) # For multiclass classification
        logger.info('Class1 precision: ' +format(sumlab1/class_labels.count(1)))
        logger.info('Class2 precision: '+format( sumlab2/class_labels.count(2)))
        logger.info('Class3 precision: '+format(sumlab3/class_labels.count(3)))
        logger.info('Class4 precision: '+format(sumlab4/class_labels.count(4)))
        logger.info('Class1 recall: ' +format(sumlab1/test_y.count(1)))
        logger.info('Class2 recall: '+format( sumlab2/test_y.count(2)))
        logger.info('Class3 recall: '+format(sumlab3/test_y.count(3)))
        logger.info('Class4 recall: '+format(sumlab4/test_y.count(4)))
        logger.info(' Misclassified 4s to class 2: '+format(mis4to2))
        logger.info(' Misclassified 2s to class 4: '+format(mis2to4))
        logger.info('  ')

        #theloume na kanoume to eswteriko ginomeno w_fixed_set me to x_t[j]
        #to w_fixed_set exei 60000 as poume mikos , enw to x_t einai se sparse morfi
        #opote theloume na dimiourgoume gia kathe j ena x mikous 60000 me midenika, 
        #kai opou exei timi to x_t, to vlepoume apo ta keys, tin prosthetoume sto x
        
        print('Time:',duration)

    '''print('Total precision for class 2 is :'+format(tot_precision_2/len(imgs)))
    print('Total precision for class 4 is :'+format(tot_precision_4/len(imgs)))
    print('Total recall for class 2 is :'+format(tot_recall_2/len(imgs)))
    print('Total recall for class 4 is :'+format(tot_recall_4/len(imgs)))'''
        
    
