import numpy as np
from sklearn import datasets
import random

f_train= open("training_data.npy","wb")
f_train_1= open("training_data.txt","w+")


f_test= open("testing_data.npy","wb")
f_test_1= open("testing_data.txt","w+")


f_av = open("average_data.npy","wb")
f_av_txt = open("average_data.txt","w+")


iris = datasets.load_iris()
data = iris.data
target = iris.target

for k in [0,1,2,3] : #...rescaling input data to [0,5]
 data[:,k] = ( ( data[:,k] - np.amin(data[:,k]) ) / ( np.amax(data[:,k])-np.amin(data[:,k]) ) )*5

training_data=np.zeros((int(len(data)/2),5))
testing_data =np.zeros((int(len(data)/2),5))

index=0
for k in [0,1,2] :
 
 split_into_two = np.split(data[np.where(target==k)],2)

 training_set = split_into_two[0]  
 testing_set  = split_into_two[1]  

 print('k=',k,'np.average(testing_set,axis=0)=',np.average(testing_set,axis=0))#,training_set) 

 for k2 in np.arange(len(training_set)) : 

    training_data[index] = np.append(training_set[k2],int(k))

   # print('training_data[k2]=',training_data[k2],'training_set[k2]=',training_set[k2])

    testing_data[index]  = np.append(testing_set[k2],int(k))

    index=index+1


#print('data=',data,'target=',target)

print('training input averages= ')
print('iris0_av=',np.average(training_data[np.where(training_data[:,4]==0)], axis=0)[0:4])
print('iris1_av=',np.average(training_data[np.where(training_data[:,4]==1)], axis=0)[0:4])
print('iris2_av=',np.average(training_data[np.where(training_data[:,4]==2)], axis=0)[0:4])

print('testing input averages= ')
print('iris0_av=',np.average(testing_data[np.where(testing_data[:,4]==0)], axis=0)[0:4])
print('iris1_av=',np.average(testing_data[np.where(testing_data[:,4]==1)], axis=0)[0:4])
print('iris2_av=',np.average(testing_data[np.where(testing_data[:,4]==2)], axis=0)[0:4])


np.random.shuffle(training_data)
np.random.shuffle(testing_data)

'''
training input averages= 
iris0_av= [1.01111111 3.08333333 0.38983051 0.30833333]
iris1_av= [2.37777778 1.61666667 2.80677966 2.59166667]
iris2_av= [3.16111111 1.93333333 3.93220339 4.05      ]

testing input averages= 
iris0_av= [0.95       2.86666667 0.39322034 0.3       ]
iris1_av= [2.16666667 1.59166667 2.71864407 2.51666667]
iris2_av= [3.19444444 2.125      3.78305085 3.975     ]

'''


for j in np.arange(len(training_data)) :

   np.save(f_train,training_data[j])
   print(training_data[j],file=f_train_1)

   np.save(f_test,testing_data[j])
   print(testing_data[j],file=f_test_1)

   
f_train.close()
f_train_1.close()

f_test.close()
f_test_1.close()






