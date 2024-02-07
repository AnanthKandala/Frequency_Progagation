import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size'   : 17})



total_epoch=1000


f4 = open("conductances_save.npy","rb")

for epoch in np.arange(total_epoch) : 

     if np.mod(epoch,10) == 0  or epoch== total_epoch - 1 :

          weights = np.load(f4)

     if epoch == 0 :
        initial_weights =  weights[:] 

     if epoch == total_epoch - 1 :
        final_weights =  weights[:] 
     
    

f_inp_nodes = open("inp_nodes.npy","rb")
f_out_nodes = open("out_nodes.npy","rb")

input_nodes = np.load(f_inp_nodes)
output_nodes = np.load(f_out_nodes)

f_inp_nodes.close()
f_out_nodes.close()


# Determine global min and max weights for consistent color mapping
global_min_weight = min(initial_weights.min(), final_weights.min())
global_max_weight = max(initial_weights.max(), final_weights.max())
   

G_initial_from_graph = nx.from_numpy_array(initial_weights)
G_final_from_graph = nx.from_numpy_array(final_weights)

# Extract weight values from the graphs for coloring
initial_weights_from_graph = nx.get_edge_attributes(G_initial_from_graph, "weight")
final_weights_from_graph = nx.get_edge_attributes(G_final_from_graph, "weight")

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 7))

# Initial Network
pos_initial = nx.spring_layout(G_initial_from_graph)
node_colors_initial = ['green' if node in input_nodes else 'red' if node in output_nodes else 'black' for node in G_initial_from_graph.nodes()]
edge_colors_initial = list(initial_weights_from_graph.values())
nx.draw(G_initial_from_graph, pos=pos_initial, with_labels=False, ax=ax1, node_size=50, 
        node_color=node_colors_initial, edge_color=edge_colors_initial, edge_cmap=plt.cm.Reds, width=2)

ax1.set_title('Initial Network')
ax1.legend(handles=[input_patch, output_patch], fontsize=15, loc='best')  # Added this line to add legend to left plot

#sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(initial_weights_from_graph.values()), vmax=max(initial_weights_from_graph.values())))
#sm.set_array([])
#plt.colorbar(sm, ax=axes[0,0])

# Create a legend for the node colors
input_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Input Nodes')
output_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Output Nodes')

# Final Network
pos_final = nx.spring_layout(G_final_from_graph)
node_colors_final = ['green' if node in input_nodes else 'red' if node in output_nodes else 'black' for node in G_final_from_graph.nodes()]
edge_colors_final = list(final_weights_from_graph.values())
nx.draw(G_final_from_graph, pos=pos_initial, with_labels=False, ax=ax2, node_size=50, 
        node_color=node_colors_final, edge_color=edge_colors_final, edge_cmap=plt.cm.Reds, width=2)
ax2.set_title('Final Network')
#sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(final_weights_from_graph.values()), vmax=max(final_weights_from_graph.values())))
#sm.set_array([])
#plt.colorbar(sm, ax=axes[0,1])

# Modify color mapping for Final Network
max_abs_weight = max(abs(global_min_weight), abs(global_max_weight))  # Calculate the maximum absolute weight
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight))  # Normalize color mapping around zero
sm.set_array([])
plt.colorbar(sm, ax=ax2)


ax2.legend(handles=[input_patch, output_patch],fontsize=15,loc='best')
plt.tight_layout()
plt.savefig("plot_single_run"+str(311)+".png", dpi=300)
plt.show()

