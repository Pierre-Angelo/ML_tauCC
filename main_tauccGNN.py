import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from tauccML_GNN import TwoGNN 

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import random
import os

if torch.cuda.is_available() :
  print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
  dev = "cuda:0"
else :
  dev = "cpu"

print(f"Device : {dev}")
device = torch.device(dev)  

def set_seed(seed = 0) :
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


# load data
dataset = 'sports' #, cstr, tr11, tr41, hitech, k1b, reviews, sports, classic3
init = 'extract_centroids' # this is the only initialization considered in the paper UNUSED

input_CSV = pd.read_csv(f'./datasets/{dataset}.txt')
target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
target = np.array(target_CSV).T[0]

table_size_x = len(input_CSV.doc.unique())
table_size_y = len(input_CSV.word.unique())
input_table = np.zeros((table_size_x,table_size_y), dtype = int)

for row in input_CSV.iterrows():
  input_table[row[1].doc,row[1].word] = row[1].cluster


# set some parameters
hidden_size = 256
embedding_size = 10
num_epochs = 100
input_dimx = table_size_x
input_dimy = table_size_y
hidden_dim = hidden_size
output_dim = embedding_size
num_layers = 2
learning_rate = 1e-4
exp_schedule = 1
threshold = 0.05
patience = 150
edge_thr = 0.2
dtype = torch.float32

print("dimensions",table_size_x,table_size_y)

# Fix seed
set_seed()
print(torch.cuda.memory_allocated(device))
data = torch.from_numpy(input_table).to(dtype).to(device)
gnn_model = TwoGNN(input_dimx, input_dimy, hidden_dim, output_dim, num_layers, learning_rate, exp_schedule, data, device)

x = data
correlation_coefficient = np.corrcoef(x.cpu())
correlation_coefficient[np.isnan(correlation_coefficient)] = 0
correlation_coefficient[correlation_coefficient < edge_thr] = 0
#correlation_coefficient = correlation_coefficient - correlation_coefficient.mean()

edge_index_x = torch.from_numpy(correlation_coefficient).nonzero().t().contiguous().to(device)
print(edge_index_x.shape)


y = data.T
correlation_coefficient = np.corrcoef(y.cpu())
correlation_coefficient[np.isnan(correlation_coefficient)] = 0
correlation_coefficient[correlation_coefficient < edge_thr] = 0
#correlation_coefficient = correlation_coefficient - correlation_coefficient.mean()

edge_index_y = torch.from_numpy(correlation_coefficient).nonzero().t().contiguous().to(device)
print(edge_index_y.shape)
print(torch.cuda.memory_allocated(device))

print("training start") 
gnn_model.fit(x, edge_index_x, y, edge_index_y, num_epochs, threshold, patience, embedding_size)

print("target :", target, "\n")

gnn_model.row_labels_ = torch.argmax(gnn_model.row_labels_, dim=1)
print("predicted row labels :", gnn_model.row_labels_,"\n")

print(f"nmi: {nmi(target, gnn_model.row_labels_.cpu())}")
print(f"ari: {ari(target, gnn_model.row_labels_.cpu())}")

## uncomment the lines below to plot tau functions

fig, ax = plt.subplots()
ax.plot(gnn_model.tau_x)
ax.plot(gnn_model.tau_y)
plt.plot([(gnn_model.tau_x[i] + gnn_model.tau_y[i])/2 for i in range(len(gnn_model.tau_x))])
ax.legend(['tau x','tau y','avg tau'])
ax.set_xlabel('iterations')
ax.set_ylabel('tau')
plt.show()
