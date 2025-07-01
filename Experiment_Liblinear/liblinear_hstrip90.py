
"""
Created on Sun Jul 17 20:18:05 2022

@author: jaime
"""

import liblinear as liblinear
import numpy as np
from liblinear.liblinearutil import *
import time
import math
import copy 

#from libsvm.svmutil import *
## READ THE DATA 

K=3
iterations=1

classes=[1, 2, 3]  

#class2: 13,33,28
#class1: 25,26,29
#class3: 13,23,26

#success combo
#filenames=['stripped25','stripped38','stripped05','stripped26','stripped39','stripped06'] #88 acc
#kaloi annotators:
#klasi 1:3,4
#klasi 2:1,5
#klasi 3:5, (3)

#most similar all classes
#filenames=['stripped13','stripped28','stripped39','stripped33','stripped23','stripped22']#,'stripped05'] #84 αcc

#manual_combo
#filenames=['stripped25','stripped28','stripped05','stripped26','stripped39','stripped22'] #88 acc

#most similar one class
#filenames=['stripped25','stripped26','stripped33','stripped28','stripped13','stripped23'] #88 acc

#mutual_info
#01
#filenames=['stripped06','stripped27','stripped02','stripped26','stripped28','stripped39']#,'stripped33','stripped25']  #89

#01 bad
#filenames=['stripped37','stripped07','stripped16','stripped05','stripped13']#,'stripped09']#,'stripped38','stripped31','stripped39']

#06
#filenames=['stripped27','stripped02','stripped25','stripped26','stripped31','stripped09']#,'stripped25'] #93 acc

#13
#filenames=['stripped27','stripped06','stripped02','stripped01','stripped28','stripped39']#,'stripped26'] #81

#13bad

#22
#filenames=['stripped06','stripped02','stripped27','stripped26','stripped25','stripped28']#,'stripped33'] #92

#31
#filenames=['stripped13']#,'stripped06','stripped25','stripped27','stripped02','stripped26','stripped09']#,'stripped38','stripped31','stripped39']#46

#33
#filenames=['stripped06','stripped27','stripped02','stripped26','stripped25','stripped28','stripped38','stripped31']#,'stripped39'] #30

#37
#filenames=['stripped06','stripped25','stripped02','stripped27','stripped31','stripped09']#,'stripped26']#,'stripped25'] #68

#39
#filenames=['stripped06','stripped27','stripped02','stripped26','stripped01','stripped28']#,'stripped38','stripped31','stripped39'] #90


#33 bad
#filenames=['stripped13','stripped05','stripped16','stripped07','stripped23','stripped22']#,'stripped38','stripped31','stripped39']

#06 hist
#filenames=['hstripped02','hstripped33','hstripped11','hstripped27','hstripped26']#,'stripped39']

#33 hist 
#filenames=['hstripped02','hstripped06','hstripped11','hstripped27','hstripped26']#,'stripped39']

#filenames=['hstripped11']
#33  class2 similarity
#filenames=['stripped25','stripped06','stripped09','stripped38','stripped05','stripped37']#,'stripped38','stripped31','stripped39']

#33 class2 class3 similarity
#filenames=['stripped25','stripped06','stripped09','stripped22','stripped28','stripped26']#,'stripped38','stripped31','stripped39']


#40
#filenames=['stripped06','stripped27','stripped02','stripped26','stripped28','stripped39']#,'stripped33','stripped25'] 

#10
#filenames=['stripped06','stripped25','stripped27','stripped37','stripped09','stripped02']#,'stripped33']#,'stripped25']

#euclidean dist

#01
filenames=['hstripp11','hstripp13','hstripp23','hstripp28','hstripp33','hstripp39']#,'stripped25']  #89

#01
#filenames=['stripped13','stripped28','stripped39','stripped33','stripped23','stripped22']#,'stripped25']  #89

#13
#filenames=['stripped39','stripped28','stripped33','stripped22','stripped23','stripped04','stripped16'] #90

#bad
#filenames=['stripped31','stripped37','stripped25','stripped06','stripped05','stripped09','stripped07']

#33 
#filenames=['stripped22','stripped38','stripped26','stripped04','stripped28','stripped07']#,'stripped38','stripped31','stripped39'] #93

#31
#filenames=['stripped13']#,'stripped06','stripped25','stripped27','stripped02','stripped26','stripped09']#,'stripped38','stripped31','stripped39']#46

#02
#filenames=['stripped26','stripped33','stripped22','stripped38','stripped06','stripped09'] #87

#05
#filenames=['stripped38','stripped25','stripped07','stripped09','stripped22','stripped26'] #84

#09
#filenames=['stripped38','stripped25','stripped26','stripped07','stripped04','stripped05'] #94

#euc mut combination

#01
#filenames=['stripped13','stripped28','stripped39','stripped06','stripped27','stripped02']#,'stripped25']  #89

#33 
#filenames=['stripped22','stripped38','stripped26','stripped06','stripped27','stripped02']#,'stripped38','stripped31','stripped39'] #93


#euclidean hsit
#01
#filenames=['hstripped13','hstripped39','hstripped28','hstripped33','hstripped23','hstripped22']#,'stripped02']  #89

#33
#filenames=['hstripped22','hstripped11','hstripped38','hstripped23','hstripped11','hstripped02']#,'stripped25']  #93

#13 
#filenames=['stripped01','hstripped39','hstripped28','hstripped33','hstripped22','hstripped23']#,'stripped25']  #89


B=len(filenames)


ulabeled_x = list()
ulabeled_y = list()  

test_x =list()
test_y = list()
      
filename='stripped01'


Y2 = list()
X2 = list()

Y2, X2 = svm_read_problem(filename)
  

ltrain_size=10000
utrain_size = 100000
test_size = 20000

ss = np.random.choice(len(Y2)-25, size=len(Y2)-25, replace=False)  #pairnoume megalutero dianysma apo oso xreiazomaste gia na apofugoume ta error logw deikth ite

for i in range(0,utrain_size):
        
    temp=dict()
    #temp[1]=X2[ss[i]][1]
    #temp[2]=X2[ss[i]][2]
    temp[3]=X2[ss[i]][3]
    temp[4]=X2[ss[i]][4]
    temp[5]=X2[ss[i]][5]
    
    temp[6]=X2[ss[i]][6]#glcm
    #temp[7]=X2[ss[i]][7]
    
    #temp[8]=X2[ss[i]][8]#glrlm
    temp[9]=X2[ss[i]][9]  
    #temp[10]=X2[ss[i]][10]
    
    #temp[11]=X2[ss[i]][11]#gldm
    temp[12]=X2[ss[i]][12]
    
    temp[13]=X2[ss[i]][13]#10perc
    temp[14]=X2[ss[i]][14]#90perc
    #temp[15]=X2[ss[i]][15] #energy
    temp[16]=X2[ss[i]][16] #mean
    
    #temp[17]=X2[ss[i]][17]  #max
    temp[18]=X2[ss[i]][18] #min
    #temp[19]=X2[ss[i]][19] #median
    
    temp[20]=X2[ss[i]][20] #rootmeansq
    #temp[21]=X2[ss[i]][21] #totalenergy
    
   
    
    
    
    ulabeled_x.append(temp)
    ulabeled_y.append(Y2[ss[i]])
        #ulabeld_y.append(copy.copy())

    
for i in range(utrain_size,utrain_size+test_size):
    
    temp=dict()
    #temp[1]=X2[ss[i]][1]
    #temp[2]=X2[ss[i]][2]
    temp[3]=X2[ss[i]][3]
    temp[4]=X2[ss[i]][4]
    temp[5]=X2[ss[i]][5]
    
    temp[6]=X2[ss[i]][6]#glcm
    #temp[7]=X2[ss[i]][7]
    
    #temp[8]=X2[ss[i]][8]#glrlm
    temp[9]=X2[ss[i]][9]  
    #temp[10]=X2[ss[i]][10]
    
    #temp[11]=X2[ss[i]][11]#gldm
    temp[12]=X2[ss[i]][12]
    
    temp[13]=X2[ss[i]][13]#firstorder 
    temp[14]=X2[ss[i]][14]
    #temp[15]=X2[ss[i]][15]  #energy
    temp[16]=X2[ss[i]][16]
    
    #temp[17]=X2[ss[i]][17]   #max
    temp[18]=X2[ss[i]][18]
    #temp[19]=X2[ss[i]][19]
    
    temp[20]=X2[ss[i]][20]
    #temp[21]=X2[ss[i]][21]
    
    
    test_x.append(temp)
    test_y.append(Y2[ss[i]])
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
    Y1, X1 = svm_read_problem(filenames[i])
    ss = np.random.choice(len(Y2)-25, size=len(Y2)-25, replace=False)  
    
    counter1= 0
    counter2= 0
    counter3= 0
    
    ite = 0
    
    w1=10
    w2=10
    w3=10
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
        
    while counter1+counter2+counter3 < ltrain_size:
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
         ite = ite + 1
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
            #temp[1]=labeled_x[j][1]
            #temp[2]=labeled_x[j][2]
            temp[3]=labeled_x[j][3]
            temp[4]=labeled_x[j][4]
            temp[5]=labeled_x[j][5]         
            
            temp[6]=labeled_x[j][6]
            #temp[7]=labeled_x[j][7]
            
            #temp[8]=labeled_x[j][8]
            temp[9]=labeled_x[j][9] 
            #temp[10]=labeled_x[j][10]
            
            #temp[11]=labeled_x[j][11]
            temp[12]=labeled_x[j][12]
            
            temp[13]=labeled_x[j][13]        
            temp[14]=labeled_x[j][14]
            #temp[15]=labeled_x[j][15] #energy
            temp[16]=labeled_x[j][16]
            #temp[17]=labeled_x[j][17] #max     
            temp[18]=labeled_x[j][18]
            #temp[19]=labeled_x[j][19]
            temp[20]=labeled_x[j][20]
            #temp[21]=labeled_x[j][21] #totalenergy
            
           
            
           
            
            sample_x.append(temp)
            sample_y.append(copy.copy(y[j])) 
        
        prob = problem(sample_y, sample_x)  #solve simple SVM for each weak anotator
        '''param = parameter('-s 2 -C ')  # na mhn pesei katw apo 10 to -c, palia 100
            
        c_params = train(prob, param)
        string= '-s 2 -c '+format(c_params[0])
        print('best_c ='+format(c_params[0]))
        param= parameter(string)'''
       
        param = parameter('-s 1 -c 4') #s=1 gia dual problem ,-C gia na vrei to kalitero c
        
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
    
#theloume na kanoume to eswteriko ginomeno w_fixed_set me to x_t[j]
#to w_fixed_set exei 60000 as poume mikos , enw to x_t einai se sparse morfi
#opote theloume na dimiourgoume gia kathe j ena x mikous 60000 me midenika, 
#kai opou exei timi to x_t, to vlepoume apo ta keys, tin prosthetoume sto x
   
duration=int(round(time.time() * 1000)) - start_time

print('Time:',duration)

