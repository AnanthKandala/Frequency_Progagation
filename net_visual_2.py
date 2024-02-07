import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20  # Set the font size to 17

total_epoch = 1000

with open("conductances_save.npy", "rb") as f4, \
        open("inp_nodes.npy", "rb") as f_inp_nodes, \
        open("out_nodes.npy", "rb") as f_out_nodes:
    
    for epoch in np.arange(total_epoch):
        if np.mod(epoch, 10) == 0 or epoch == total_epoch - 1:
            weights = np.load(f4)
        if epoch == 0:
            initial_weights = weights[:]
        if epoch == total_epoch - 1:
            final_weights = weights[:]
    
    input_nodes = np.load(f_inp_nodes)
    output_nodes = np.load(f_out_nodes)

change_in_weights = final_weights - initial_weights
G_initial = nx.from_numpy_array(initial_weights)
G_change_in_weights = nx.from_numpy_array(change_in_weights)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Initial Network
pos_initial = nx.spring_layout(G_initial)
node_colors_initial = ['green' if node in input_nodes else 'red' if node in output_nodes else 'black' for node in G_initial.nodes()]
edge_colors_initial = [initial_weights[u, v] for u, v in G_initial.edges()]
nx.draw(G_initial, pos=pos_initial, with_labels=False, ax=axs[0], node_size=50,
        node_color=node_colors_initial, edge_color=edge_colors_initial, edge_cmap=plt.cm.Blues, width=2)
axs[0].set_title('Initial Network')
sm_initial = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=np.min(initial_weights), vmax=np.max(initial_weights)))
sm_initial.set_array([])
cbar_initial = plt.colorbar(sm_initial, ax=axs[0])
cbar_initial.set_label('Initial Weights', rotation=270, labelpad=15)
for label in cbar_initial.ax.yaxis.get_ticklabels():
    label.set_weight("bold")  # Set ticklabel to bold for Initial Weights colorbar

# Changes in Network Weights
#pos_change = nx.spring_layout(G_change_in_weights)
node_colors_change = ['green' if node in input_nodes else 'red' if node in output_nodes else 'black' for node in G_change_in_weights.nodes()]
edge_colors_change = [change_in_weights[u, v] for u, v in G_change_in_weights.edges()]
nx.draw(G_change_in_weights, pos=pos_initial, with_labels=False, ax=axs[1], node_size=50,
        node_color=node_colors_change, edge_color=edge_colors_change, edge_cmap=plt.cm.RdYlBu, width=2)
axs[1].set_title('Change in Network Weights')

max_abs_change = np.max(np.abs(change_in_weights))  # Find maximum absolute value in change_in_weights
sm_change = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(vmin=-max_abs_change, vmax=max_abs_change))  # Normalize color map 
sm_change.set_array([])
cbar_change = plt.colorbar(sm_change, ax=axs[1])
cbar_change.set_label('Change in Weights', rotation=270, labelpad=15)



for label in cbar_change.ax.yaxis.get_ticklabels():
    label.set_weight("bold")  # Set ticklabel to bold for Change in Weights colorbar

input_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Input Nodes')
output_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Output Nodes')
axs[1].legend(handles=[input_patch, output_patch], fontsize=15, loc='best')
axs[0].legend(handles=[input_patch, output_patch], fontsize=15, loc='best')  # Add legend to axs[0]

plt.tight_layout()
plt.savefig("initial_and_change_in_weights_plot.png", dpi=300)
plt.show()

