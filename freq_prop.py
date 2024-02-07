#!/usr/bin/env python
import numpy as np
from math import *
import sys
import matplotlib.pyplot as plt
import networkx as nx

def degree_mat(weight_matrix) :

    degree_matrix = np.zeros((len(v),len(v)))
    for i in np.arange(len(v)) : 
       degree_matrix[i,i] = np.sum(weight_matrix[i,:]) 
       # print('non zeros =',i,np.where(w[i,:]!=0),len(np.where(w[i,:]!=0)[0]) )
       for j in np.arange(len(v)) :
  
          if np.transpose(weight_matrix[i,j]) != weight_matrix[i,j] :
            sys.exit('the weight matrix is not symmetric',np.transpose(weight_matrix[i,j]), '!=', weight_matrix[i,j])

    return degree_matrix




def desired_voltage_drop() :

    vdesired = np.zeros((no_classes,no_out))
   
   
    print('vdesired=',vdesired)

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
         in_data =  np.array([1.01111111 , 3.08333333, 0.38983051, 0.30833333])
      if i == 1 :
         in_data =  np.array([2.37777778, 1.61666667, 2.80677966 ,2.59166667])
      if i == 2 :
         in_data =  np.array([3.16111111, 1.93333333, 3.93220339, 4.05      ])    

      for j34 in np.arange(no_inp) :
         
         print(j34,2*j34)
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

      

def update_matrix(ff_chem,fb_chem,weight_matrix,learning_rate) : # given potential at every vertex x, this gives an array of potential drops

    dw = np.zeros((len(v),len(v)))

    new_wt = np.zeros((len(v),len(v)))

    for i in np.arange(len(v)) :
      for j in np.where(weight_matrix[i,:] != 0)[0] :
         
         if j > i : #..because matrix is symmetric, we dont have to update two times

           change = learning_rate*(ff_chem[i] - ff_chem[j])*(fb_chem[i] - fb_chem[j])

           dw[i,j] = change 
           dw[j,i] = change

           if ( weight_matrix[i,j] + change ) <= 0 :

              new_wt[i,j] = min_weight
              new_wt[j,i] = min_weight

              min_weight_matrix[i,j] = min_weight_matrix[i,j] + 1 # save no of times ij reaches minimum
              min_weight_matrix[j,i] = min_weight_matrix[j,i] + 1

           elif  ( weight_matrix[i,j] + change ) > max_weight :

              new_wt[i,j] = max_weight
              new_wt[j,i] = max_weight

              max_weight_matrix[i,j] = max_weight_matrix[i,j] + 1
              max_weight_matrix[j,i] = max_weight_matrix[j,i] + 1

           else :
         
              new_wt[i,j] = weight_matrix[i,j]  + change
              new_wt[j,i] = weight_matrix[i,j]  + change

          
    return new_wt , dw
 
def output_node_pairer(single_output_node) : # generates output node set that are composed of neighbouring pairs

    output_node = np.array([])

    for i in single_output_node :

        available_set = np.where(w[i,:] != 0)[0] #..set of neighbours of i
        pair = np.random.choice( available_set, 1 ,replace=False ) #... choose one from the set of neighbours

        output_node = np.append(output_node,i)
        output_node = np.append(output_node,pair)

    return output_node.astype(int)



def add_edges(w_old,edges_to_add) : # adds entries at empty (i,j) and (j,i) positions of a weight matrix

    
    arr=np.array(np.column_stack(np.where(w_old == 0)))
 #   ('arr=',arr)
    np.random.shuffle(arr)
  #  print('w_old=',w_old,'arr_shuf=',arr)

    warr_covered = {}
    edges_added = 0
    
    for warr in arr :
      if edges_added  < edges_to_add :
      #  print('warr=',warr)
        if (({warr[0],warr[1]} in warr_covered.values()) or (warr[0]==warr[1])) == False :

            warr_covered[len(warr_covered)] = {warr[0],warr[1]}

        #    print('w_old=',w_old)
            w_old[warr[0],warr[1]] = 1
            w_old[warr[1],warr[0]] = 1

            edges_added = edges_added + 1
        #    print( 'warr_covered=', warr_covered, 'w_old=',w_old,'edges_added=',edges_added)

    
  
    return w_old



def random_update_matrix(weight_matrix) : # given potential at every vertex x, this gives an array of potential drops


    new_wt = np.zeros((len(v),len(v)))

   

    for i in np.arange(len(v)) :
      for j in np.where(weight_matrix[i,:] != 0)[0] :
         
         if j > i : #..because matrix is symmetric, we dont have to update two times

           new_cond =  np.random.uniform(min_weight,max_weight) # (1e-8)*noise_amp*np.random.normal(0,1)
 

           new_wt[i,j] = new_cond
           new_wt[j,i] = new_cond

             

          
    return new_wt 



#............. PROGRAM STRART

#.....(part A start) 

# A1 : Define the network architecture 

np.set_printoptions(threshold=sys.maxsize)

n_tot =  40
edges_to_add = 200


max_weight     = 1
min_weight     = 0.00001


v=np.arange(n_tot) #...index the vertices


G=nx.barabasi_albert_graph(n_tot,1)
          #........................GET RELEVANT MATRICES............................................... 

# A2 : Get relevant matrices of the network  

w= random_update_matrix(add_edges(np.array(nx.adjacency_matrix(G).todense()),edges_to_add)) 
 
d = degree_mat(w)         # get the degree matrix,degree of i= sum of the weights of egdes connected to i
l=d-w                     # get the laplacian matrix

# (part A end)....................................................

#.....save fixed system parameters...
f36 = open("vertices_save.npy","wb")
np.save(f36,v)
f36.close()

#.......save dynamic system parameters at the end of every instance

#f34_ff = open("ff_voltages_save.npy","wb")
#f34_fb = open("fb_voltages_save.npy","wb")

f35 = open("conductances_save.npy","wb")
f_tune=open("tuning_data.npy","wb")
f_output_voltages=open("des_output_volt.npy","wb")

f_inp_nodes = open("inp_nodes.npy","wb")
f_out_nodes = open("out_nodes.npy","wb")

# (Part B Start)  1. define relevant nodes 2. define feedback and feedforward chemical..........................................


# 1B ......................... define input output nodes.. 

no_inp = 4     #   no of input variables
no_out = 3   #   no of output variables
no_classes = 3 #   no of classes of the classificaton problem

max_weight = 20
min_weight = 0.00001

min_weight_matrix = np.zeros((len(v),len(v))) #... i,j entry shows no of times the edge ij reached min value
max_weight_matrix = np.zeros((len(v),len(v))) #... i,j entry shows no of times the edge ij reached min value


# for each input we need two nodes, similarly for output. 

relevant_nodes = np.random.choice( np.arange(len(v)), 2*(no_inp + no_out) ,replace=False ) #..for no pair
#relevant_nodes = np.random.choice( np.arange(len(v)), 2*(no_inp) + no_out ,replace=False ) #.for pair

inp_nodes  =    relevant_nodes[0:(2*no_inp)]   # np.array([34, 35 , 3 ,52 , 44, 21 ,  6, 61]) .. where input data is given as currents


out_nodes = relevant_nodes[(2*no_inp):(2*no_inp)+(2*no_out) ] #..for no pair

#out_nodes = output_node_pairer(relevant_nodes[(2*no_inp):(2*no_inp)+(no_out) ])     #  np.array([23, 24 , 22, 30, 43, 57 ]) .. where output is measured as potential drop across two nodes . Here output nodes are neighbouring nodes

'''
I will follow this convention :- first two is one pair, second two is another pair and so on.
                                 Out of the pair the node that comes first gets positive current
''' 
int_nodes = np.setdiff1d(np.arange(len(v)),inp_nodes)   

print('relevant_nodes=',relevant_nodes,'inp_nodes=',inp_nodes,'out_nodes=',out_nodes,'int_nodes=',int_nodes)

np.save(f_inp_nodes,inp_nodes)
np.save(f_out_nodes,out_nodes)

f_inp_nodes.close()
f_out_nodes.close()         

# 2B

sv = np.zeros(len(v)) #... sv is feedforward chemical
su = np.zeros(len(v)) #... su is feedback chemical


#........freq_prop_time
one_step_time = 5
dt = 0.05
omega = (2.0)*np.pi # period is 1 , one_step_time is number of periods we consider
len_series = len(np.arange(0,one_step_time,dt))
frequency = omega / (2 * np.pi)  # convert angular frequency to frequency
sampling_rate = 1 / dt  # dt is the time step
index_of_fft = int(round(frequency * len_series / sampling_rate))  # find the index corresponding to the frequency

sine_series = np.array([ sin(omega*t) for t in np.arange(0,one_step_time,dt) ])

# (Part C start )


eta = 1 #..nudge_amplitude
learning_rate = 0.01
total_epoch = 1000
for epoch in np.arange(total_epoch) : 

     f_input=open("training_data.npy","rb")
     vdesired = desired_voltage_drop()
     if np.mod(epoch,10) == 0 or epoch== total_epoch - 1 :
         np.save(f35,w)
     np.save(f_output_voltages,vdesired)

     for instance in np.arange(75) : # here 100 is size of data set

            #...................... load training data..get input data and desired output data
            incoming_data = np.load(f_input)

            input_data = incoming_data[0: no_inp] #..... data inputed in the network as current applied on input node pair 
            current_target_iris = int(incoming_data[4])

            desired_out =  vdesired[current_target_iris,:] #..........this is the potential drop we want at the output node pair
            
            remaining_iris= np.setdiff1d(np.array([0,1,2]),np.array([current_target_iris])) 
            #...........................




        

          
            #........ get corresponding output feedforward voltage drops for the given input data
            iext = np.zeros(len(v))
            present_out = np.zeros(no_out) 

            for j67 in np.arange(no_inp) :
         
               
                iext[inp_nodes[2*j67]]      =  input_data[j67] # give input data as applied current on the input node pairs
                iext[inp_nodes[(2*j67)+1]]  = -input_data[j67] 

         #   print('input_data=',input_data,'iext=',iext)

            #solving for voltages using laplacian...pseudoinverse of laplacian is used because the laplacian is noninvertible
            inf_v = np.matmul(np.linalg.pinv(l,rcond=1e-5,hermitian=True),iext) #..get the voltages at each vertex due to external current
         #   print('feedforward_voltages=',vol,'inp_nodes=',inp_nodes,'out_nodes=',out_nodes,'feedforward_voltages[out_nodes]=',v[out_nodes])

            for j23 in np.arange(no_out) :

               present_out[j23] = inf_v[out_nodes[2*j23]]-inf_v[out_nodes[(2*j23)+1]]
            
            #.................................... apply ac and dc frequencies
            estimated_a = []
            estimated_b = []
            time_series = []


           # eext_dc = np.zeros(len(v))
          #  for j34 in np.arange(no_out) :
         
          #          eext_dc[out_nodes[2*j34]]      =  eta*(present_out[j34]-desired_out[j34])
         #           eext_dc[out_nodes[(2*j34)+1]]  = -eta*(present_out[j34]-desired_out[j34])

        #    su = np.matmul(np.linalg.pinv(l,rcond=1e-5,hermitian=True),eext_dc)
            
     
            eext = np.zeros(len(v))
          
            for t in np.arange(0,one_step_time,dt) :

               
                for j34 in np.arange(no_out) :
         
                    eext[out_nodes[2*j34]]      =  eta*(present_out[j34]-desired_out[j34])*sin(omega*t)
                    eext[out_nodes[(2*j34)+1]]  = -eta*(present_out[j34]-desired_out[j34])*sin(omega*t) 

                print('epoch=',epoch,'instance=',instance,'t=',t)

                time_series.append(np.matmul(np.linalg.pinv(l,rcond=1e-5,hermitian=True),(iext+eext) ))  # Appendthe time series list for this instance
                
            time_series_array = np.array(time_series)

            for series in time_series_array.T:  # For each individual series in the time series array

           
                a0 = np.mean(series)
                b1 = 2*np.mean(series*sine_series)
            
                estimated_a.append(a0)
                estimated_b.append(b1)


            
        #    print('inf_v=',inf_v-inf_v[0])
        #    print('estimated_a=',estimated_a-estimated_a[0])
        #    print('su=',su-su[0])
        #    print('estimated_b=',estimated_b-estimated_b[0])
           
                
 
            #........... update weights and get new laplacian
            
            w,dw = update_matrix(estimated_a,estimated_b,w,learning_rate)   
        
            d = degree_mat(w)         # updated degree matrix
            l=d-w                     # updated laplacian matrix
            #..............................................................................





            #...........save relevant training data
  
            error = np.linalg.norm( present_out - desired_out )

            error_normalizer = np.linalg.norm( present_out - vdesired[remaining_iris[0],:] ) + np.linalg.norm( present_out - vdesired[remaining_iris[1],:] ) 

            normalized_error = (error/error_normalizer)

            conductance_change = np.linalg.norm(dw)

            print('epoch=',epoch,'instance=',instance,'current_target_iris=',current_target_iris)
            print('present_out=',present_out,'desired_out=',desired_out)
           
            print( 'error=', error,'normalized_error=',normalized_error)  
            print(' ')
   
            #.........these details are saved at the end of every instance
            instance_details=np.array([epoch,instance,error,normalized_error,conductance_change])
            np.save(f_tune,instance_details)


     #...................,block change, saving after each epoch

     
     f_input.close() #...close the input data file at the end of each epoch

f_tune.close()
#f34_ff.close()
#f34_fb.close()
f35.close()
f_output_voltages.close()

f_min_wt = open("min_wt_save.npy","wb")
f_max_wt = open("max_wt_save.npy","wb")

np.save(f_min_wt,min_weight_matrix)
np.save(f_max_wt,max_weight_matrix)

f_min_wt.close()
f_max_wt.close()

print('inp_nodes=',inp_nodes,'out_nodes=',out_nodes)













