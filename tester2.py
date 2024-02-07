#!/usr/bin/env python

"""
 1. The code creates an 2d square lattice of size n_tot*n_tot and distorts it
 2. Then the connections are made with n nearest neighbours ( n= degree), each connection is assigned a random conductance
 3. using conductance matrix and degree matrix , laplacian matrix for the graph is defined
 4. using the laplacian the potentials are calculated

   Apr,2021

 1. We now want our network to give a desired output for a given input

 2. We will encode the input as the "deviations of potentials" applied on bnd nodes. Here large fixed potentials are always applied on boundary nodes. This is because the min and max of potentials occurs on boundary nodes, therefore we should make sure that the output potentials are within this range


 3. we will input the desired output as the "deviations of potentials" on the output nodes of the network 

"""


import numpy as np
from math import *
import sys
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 17  # Set the font size to 17


def conductance_matrix(v) : # make connections between the close neighbours and assign a random conductance value between 0 and 1
    
    rm = np.zeros((len(v),len(v))) # conductance matrix

    for i in np.arange(len(v)) :

      dist_matrix = np.zeros(len(v))
      for j in np.arange(len(v)) : 

        dist = sqrt((v[i,0]-v[j,0])**2 + (v[i,1]-v[j,1])**2) 
        if  min_connection_radius < dist < max_connection_radius : # points too close and too far are not considered for connection
           dist_matrix[j] = dist
        else :
           dist_matrix[j] = np.inf # points too close or too far are enetered as np.inf's, it also avoids self connection. preserving the degree

      min_indices = dist_matrix.argsort()[:degree] # only n closest no of points are chosen for connection, n=degree
   
      min_indices_tempo = np.array([])
      for l in np.arange(len(min_indices)) :
          if dist_matrix[min_indices[l]] != np.inf : # we have to make sure min_indices dont have any indices that correspond to np.inf in dist_matrix
              min_indices_tempo = np.append(min_indices_tempo,min_indices[l])

      min_indices=min_indices_tempo.astype(int)
   
      for k in np.arange(len(min_indices)) : 

       cond = 1.0 # np.random.uniform(0,10) # assign a random conductance to the link between 0 and 10
       rm[i,min_indices[k]] = cond  # same random value is assigned to (i,j) and (j,i) as the graph is undirected.
       rm[min_indices[k],i] = cond  # delete this line if you want a directed graph

    return rm

def degree_mat(weight_matrix) :

    degree_matrix = np.zeros((len(v),len(v)))
    for i in np.arange(len(v)) : 
       degree_matrix[i,i] = np.sum(w[i,:]) 
       # print('non zeros =',i,np.where(w[i,:]!=0),len(np.where(w[i,:]!=0)[0]) )
       for j in np.arange(len(v)) :
  
          if np.transpose(w[i,j]) != w[i,j] :
            sys.exit('the weight matrix is not symmetric',np.transpose(w[i,j]), '!=', w[i,j])

    return degree_matrix
 

def current_plot() : #.............plotting current direction. basically an arrow is drawn from high to low potential

  for i2 in np.arange(len(v)) :
   for j2 in np.arange(i2,len(v)) :
     if w[i2,j2] != 0 :
       ax.plot([v[i2,0],v[j2,0]],[v[i2,1],v[j2,1]],linewidth = w[i2,j2]/10,color='b')
       if voltages[i2] > voltages[j2] :
          plt.arrow(v[i2,0],v[i2,1],(v[j2,0]-v[i2,0])/2,(v[j2,1]-v[i2,1])/2 ,color='b', shape='full', lw=0, length_includes_head=True, head_width=0.5)
       elif voltages[i2] < voltages[j2]  :
          plt.arrow(v[j2,0],v[j2,1],(v[i2,0]-v[j2,0])/2,(v[i2,1]-v[j2,1])/2 ,color='b', shape='full', lw=0, length_includes_head=True, head_width=0.5)
       elif voltages[i2] == voltages[j2] :
          ax.plot([v[i2,0],v[j2,0]],[v[i2,1],v[j2,1]],linewidth = w[i2,j2]/10,color='r')

  volt_plot=ax.scatter(v[:,0],v[:,1],s=100,c=voltages, cmap='rainbow', alpha=10)
 
  for i67 in output_vertices :
    ax.scatter(v[i67,0],v[i67,1],s=voltages[i67]*80,marker='x',lw=3,color='g') # label='current source'
  for i78 in input_vertices :
    ax.scatter(v[i78,0],v[i78,1],s=voltages[i78]*80,marker='x',color='r',lw=3) #,label='current sink'

  ax.annotate("Thickness of connection represents conductance values, crosses denote current sink and source ",(1,97))

  cbar=plt.colorbar(volt_plot)
  cbar.set_label('voltages')

  return cbar


def node_indices(layer) :

    first_index = np.sum(npl[0:layer])
    
    node_index = np.array([])

    for i in np.arange(npl[layer]) :

      node_index = np.append(node_index,first_index+i)


    node_index = node_index.astype(int)

    return node_index       

def response_matrices(lap,bnd_n,int_n) :

    x_mat = lap[bnd_n,:]
    a_mat = x_mat[:,bnd_n]
    b_mat = x_mat[:,int_n]

    y_mat = lap[int_n,:]
    d_mat = y_mat[:,int_n]

    mat1 = -np.matmul(np.linalg.inv(d_mat),np.transpose(b_mat)) # mat1= - (D^-1)*(B^T)

    mat2 = a_mat + np.matmul(b_mat,mat1) # mat2= A - (B)*(D^-1)*(B^T)

    return mat1,mat2
   

def int_poten(volt,bnd_n,int_n,mat1,mat2) :

   # print('initial_voltages=',volt)

    iext_arr =np.zeros(len(v))

    int_pot = np.matmul(mat1,volt[bnd_n])

    currents = np.matmul( mat2 , volt[bnd_n] )

    volt[int_n] = int_pot 

  #  print('currents=',currents)

    iext_arr[bnd_n] = currents
    
  #  print('int_pot=',int_pot,'pot[int_n]=',volt[int_n],'voltages=',voltages)
  
    return volt,iext_arr

def desired_voltage_drop() :

    vdesired = np.zeros((no_classes,no_out))
   
   
   
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
    
   
    for i in [0,1,2] :

      iext = np.zeros(len(v))

      if i == 0 :
         in_data =  np.array([1.01111111, 3.08333333, 0.38983051, 0.30833333])
      if i == 1 :
         in_data =  np.array([2.37777778 ,1.61666667, 2.80677966, 2.59166667])
      if i == 2 :
         in_data =  np.array([3.16111111, 1.93333333, 3.93220339, 4.05      ])    

      for j34 in np.arange(no_inp) :
         
         iext[inp_nodes[2*j34]]      =  in_data[j34] # give input data as applied current on the input node pairs
         iext[inp_nodes[(2*j34)+1]]  = -in_data[j34] 

    #  print('iext=',iext)

      #solving for voltages using laplacian...pseudoinverse of laplacian is used because the laplacian is always noninvertible
      voltages = np.matmul(np.linalg.pinv(l,rcond=1e-5,hermitian=True),iext) #..get the voltages at each vertex due to external current
   #   print('voltages=',voltages,'out_nodes=',out_nodes,'voltages[out_nodes]=',voltages[out_nodes])

      for j67 in np.arange(no_out) :

        vdesired[i,j67] = voltages[out_nodes[2*j67]]-voltages[out_nodes[(2*j67)+1]]
    #    print('vdesired=',vdesired)


    return vdesired

      




#.............


total_epoch = 1000
gap = 1 # testing is done after every 'gap' epoch
total_instances = 75
total_time=total_epoch*total_instances

f5 = open("vertices_save.npy","rb")
v= np.load(f5)
f5.close()



#........load input output vertices

no_inp=4 # no of input nodes and no of output nodes
no_out=3
no_classes=3

f_inp_nodes = open("inp_nodes.npy","rb")
f_out_nodes = open("out_nodes.npy","rb")

inp_nodes = np.load(f_inp_nodes)
out_nodes = np.load(f_out_nodes)

f_inp_nodes.close()
f_out_nodes.close()

int_nodes = np.setdiff1d(np.arange(len(v)),inp_nodes)   
#....................................................

f4 = open("conductances_save.npy","rb")

accuracy_arr = np.array([])
wrong_arr = np.array([])
x_arr = np.array([])

for epoch in np.arange(total_epoch) :

  if np.mod(epoch,10) == 0 or epoch== total_epoch - 1 :
    
 #.................................................................. 
     w= np.load(f4) # get the conductnce matrix
     d = degree_mat(w)         # get the degree matrix,degree of i= sum of the weights of egdes connected to i
     l=d-w                     # get the laplacian matrix
 #....................................................

     vdesired = desired_voltage_drop()  

     f_input=open("testing_data.npy","rb")
     correct_prediction = 0 
     wrong_prediction = 0
     correction_saver = np.zeros((3,2)) # saves which data was correct and which was not :- [#right,#wrong], ith row denotes its output

     instance = 0
     while instance < total_instances and np.mod(epoch,gap) == 0 : # here 100 is size of data set

 
  #...................... load training data..get input data and desired output data
      incoming_data = np.load(f_input)

      input_data = incoming_data[0: no_inp] #..... data inputed in the network as current applied on input node pair 
      desired_target_iris = int(incoming_data[4])

      desired_out =  vdesired[desired_target_iris,:] #..........this is the potential drop we want at the output node pair
            
      remaining_iris= np.setdiff1d(np.array([0,1,2]),np.array([desired_target_iris])) 
  #...........................




  #........ get corresponding output feedforward voltage drops for the given input data
      iext = np.zeros(len(v))
      present_out = np.zeros(no_out) 

      for j67 in np.arange(no_inp) :
         
               
           iext[inp_nodes[2*j67]]      =  input_data[j67] # give input data as applied current on the input node pairs
           iext[inp_nodes[(2*j67)+1]]  = -input_data[j67] 

         
  #solving for voltages using laplacian...pseudoinverse of laplacian is used because the laplacian is noninvertible
      sv = np.matmul(np.linalg.pinv(l,rcond=1e-5,hermitian=True),iext) #..get the voltages at each vertex due to external current
  #   print('feedforward_voltages=',sv,'inp_nodes=',inp_nodes,'out_nodes=',out_nodes,'feedforward_voltages[out_nodes]=',sv[out_nodes])

      for j23 in np.arange(no_out) :

            present_out[j23] = sv[out_nodes[2*j23]]-sv[out_nodes[(2*j23)+1]]
  #....................................




  #.......canculate the distances between present output and desired output for every class
      dist=np.zeros(3)

      for i in [0,1,2] :
    
         diff = present_out - vdesired[i]
         dist[i] = np.linalg.norm(diff)

         print('class=',i,'present_out=',present_out,'vdesired[class]=',vdesired[i],'diff=',diff,'dist=',dist)
     

      out_index = np.where(dist==np.amin(dist))[0][0] #........choose the class where the desired output is closest to present output
  

      print('instance=',instance,'output_target_iris=',out_index,'desired_target_iris=',desired_target_iris )
      print(' ')

      if np.array_equal(out_index,desired_target_iris) == True :
         correct_prediction=correct_prediction+1
         correction_saver[out_index,0] = correction_saver[out_index,0] + 1

      if np.array_equal(out_index,desired_target_iris) == False :
         wrong_prediction=  wrong_prediction+1
         correction_saver[out_index,1] = correction_saver[out_index,1] + 1

 
      print( 'correct_prediction = ',correct_prediction , 'wrong_prediction = ',wrong_prediction)
      print('right,wrong')
      print('correction_saver=',correction_saver)

      if correct_prediction != 0 :
 
        accuracy= (correct_prediction)/(wrong_prediction+correct_prediction)  # old wrong :(correct_prediction-wrong_prediction)/correct_prediction
  
      instance = instance + 1
  
     if np.mod(epoch,gap) == 0 :

      print('epoch=',epoch,'accuracy=', accuracy )

      accuracy_arr  = np.append(accuracy_arr,accuracy)
      x_arr = np.append(x_arr,epoch)
      wrong_arr  = np.append(wrong_arr,wrong_prediction)
 
     f_input.close()

f4.close()

fig = plt.figure(figsize=(10,20))

ax1 = fig.add_subplot(1, 1, 1)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15) # define plotting area

#ax2 = fig.add_subplot(1, 2, 2)

ax1.set_xlabel("Epoch",labelpad=0,size=30)

ax1.set_ylabel("Accuracy",labelpad=0,size=30)

#ax1.set_xticks([0, 50,100, 150,200,250])*4
'''
ax1.xaxis.set_tick_params(width=5,length=10,direction="out")

ax1.yaxis.set_tick_params(width=5,length=10,direction="out")

for axis in ['top','bottom','left','right']: # set width of axis

    ax1.spines[axis].set_linewidth(1)

for tick in ax1.xaxis.get_major_ticks(): # set size of tick labels

                tick.label.set_fontsize(30)

for tick in ax1.yaxis.get_major_ticks(): # set size of tick labels

                tick.label.set_fontsize(30)

print('x_arr=',x_arr,'accuracy_arr=',accuracy_arr)
'''
ax1.plot(x_arr,accuracy_arr,lw=4)

#ax1.set_xscale('log')
#ax1.set_yscale('log')

plt.savefig("accuracy_plot.png", dpi=300)

plt.show()

    
