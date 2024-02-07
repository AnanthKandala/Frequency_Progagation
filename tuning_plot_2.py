import numpy as np
import matplotlib.pyplot as plt



plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 17  # Set the font size to 17

f22= open("tuning_data.npy","rb")

#instance_details=np.array([epoch,instance,error,normalized_error,conductance_change])

epoch=np.array([])
instance,error,nor_error=np.array([]),np.array([]),np.array([])
cond_chg,spread=np.array([]),np.array([])




total_epoch = 1000
examples_per_epoch=75
total_no = total_epoch*examples_per_epoch

gap=1
for i in np.arange(total_no ) :

   
    tuning_data = np.load(f22)
    print((i/total_no)*100) 
    if np.mod(i,gap) == 0 :

      epoch = np.append(instance,tuning_data[0])
      instance=np.append(instance,tuning_data[1])
      error=np.append(error,tuning_data[2])
      nor_error=np.append(nor_error,tuning_data[3])
      cond_chg = np.append(cond_chg,tuning_data[4])
     
f22.close()

fig = plt.figure(figsize=(10,20))

plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size'   : 20})


ed=error

delta= examples_per_epoch # averaging window

av_error=np.zeros(int(len(ed)/delta))
std_error=np.zeros(int(len(ed)/delta))

av_nor_error=np.zeros(int(len(ed)/delta))
std_nor_error=np.zeros(int(len(ed)/delta))

av_cond_chg=np.zeros(int(len(ed)/delta))
std_cond_chg=np.zeros(int(len(ed)/delta))


av_spread=np.zeros(int(len(ed)/delta))
std_spread=np.zeros(int(len(ed)/delta))


for j in np.arange(int(len(ed)/delta)) :

  av_error[j] = np.mean(error[j*delta:(j+1)*delta])
  std_error[j] = np.std(error[j*delta:(j+1)*delta])

  av_nor_error[j] = np.mean(nor_error[j*delta:(j+1)*delta])
  std_nor_error[j] = np.std(nor_error[j*delta:(j+1)*delta])

  av_cond_chg[j]  = np.mean(cond_chg[j*delta:(j+1)*delta])
  std_cond_chg[j] = np.std(cond_chg[j*delta:(j+1)*delta])

  av_spread[j] = np.mean(spread[j*delta:(j+1)*delta])
  std_spread[j] = np.std(spread[j*delta:(j+1)*delta])


x_axis = np.arange(0,total_epoch,1/examples_per_epoch)

av_axis = np.arange(0,total_epoch)
print('len(av_axis)=',len(av_axis),'len(av_error)=',len(av_error),av_axis)

#fig.suptitle("Epoch=250,Instances=75,#nodes=12*12,eta=1e-4,w~(mean=0.1,sd=0.01)")

print('len(av_axis)=',len(av_axis),'len(av_error)=',len(av_error),av_axis)

ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(1, 2, 2)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15) # define plotting area


print('len(av_axis)=',len(av_axis),'len(cond_chg)=',len(cond_chg))


ax1.errorbar(av_axis,av_error,yerr=std_error,color='red',ecolor='red',alpha=1,lw=4,elinewidth=2,capsize=2,errorevery=25,label='Average over one epoch',zorder=2)
#ax1.plot(av_axis,av_error)
#ax1.set_ylim(0,0.08)
#ax1.set_title('Error')
ax1.plot(x_axis,error,'*',alpha=0.1,zorder=1)
#x_ticks = np.arange(0,100)
#print('xticks=',x_ticks)
#ax1.set_xticks(x_ticks)

ax1.set_xlabel("Epoch",labelpad=0,size=30)

ax1.set_ylabel("Error",labelpad=0,size=30)

#ax1.set_xticks([0, 50,100, 150,200,250])*4

ax1.xaxis.set_tick_params(width=5,length=10,direction="out")

ax1.yaxis.set_tick_params(width=5,length=10,direction="out")

'''
ax2.plot(x_axis,cond_chg,'*',alpha=0.1,zorder=1)
ax2.errorbar(av_axis,av_cond_chg,yerr=std_cond_chg,color='red',ecolor='yellow',alpha=1,lw=4,elinewidth=1,capsize=1,errorevery=25,label='Average over one epoch',zorder=2)
#ax2.set_ylim(0,0.08)
#ax2.set_title('Normalized Error')
#ax2.set_xticks(x_ticks)

ax2.set_xlabel("Epoch",labelpad=0,size=30)

ax2.set_ylabel("Step size",labelpad=0,size=30)

ax2.set_xticks([0, 50, 100, 150, 200,250])

ax2.xaxis.set_tick_params(width=5,length=10,direction="out")

ax2.yaxis.set_tick_params(width=5,length=10,direction="out")




ax2.plot(x_axis,cond_chg,'*',alpha=0.1)
ax2.errorbar(av_axis,av_cond_chg,yerr=std_cond_chg,color='red',ecolor='green',elinewidth=1,capsize=2,errorevery=100,label='Average over one epoch')
#ax2.set_ylim(0,0.08)
ax2.set_title('Conductance change')
#ax2.set_xticks(x_ticks)
ax2.set_xlabel("Epoch")
'''

for axis in ['top','bottom','left','right']: # set width of axis

    ax1.spines[axis].set_linewidth(3)

for tick in ax1.xaxis.get_major_ticks(): # set size of tick labels

                tick.label.set_fontsize(30)

for tick in ax1.yaxis.get_major_ticks(): # set size of tick labels

                tick.label.set_fontsize(30)


#ax1.set_xscale('log')
#ax1.set_yscale('log')
plt.savefig("training_plot.png", dpi=300)
plt.legend()
plt.show()

   





