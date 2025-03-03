import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from tauccML_GNN import TwoGNN 

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def adj_correlation(data,percentile = 90):
  adj = np.corrcoef(data)
  adj[np.isnan(adj)] = 0
  adj[adj < np.percentile(adj,percentile)] = 0

  
  return adj

def adj_cooccurence(data, percentile = 90):
  adj = np.matmul(data,data.T)
  np.fill_diagonal(adj,0)
  adj[adj < np.percentile(adj,percentile)] = 0

  return adj

# load data
dataset = 'cstr' # cstr, tr11, classic3, hitech, k1b, reviews, sports, tr41

target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
target = np.array(target_CSV).T[0]

input_table = np.load(f'./data/{dataset}.npy')
input_dimx, input_dimy = input_table.shape

# Parameters
hidden_dim = 128
explained_variance = 0.8
embedding_size = 10
num_epochs = 100
num_layers = 2
learning_rate = 1e-3
exp_schedule = 1
threshold = 0.1
patience = 20
edge_percentile = 95 
dtype = torch.float32

# Fix seed
set_seed()

# Generate feature vector and adjacency matrix
objects_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_x_0.8.npy')).to(dtype).to(device)
objects_edge_index = torch.from_numpy(adj_correlation(input_table,edge_percentile)).nonzero().t().contiguous().to(device)

features_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_y_0.8.npy')).to(dtype).to(device)
features_edge_index = torch.from_numpy(adj_correlation(input_table.T,edge_percentile)).nonzero().t().contiguous().to(device)

data = torch.from_numpy(input_table).to(dtype).to(device)
gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], hidden_dim, embedding_size, num_layers, learning_rate, exp_schedule, data, device)


print("training start") 
gnn_model.fit(objects_embedding, objects_edge_index, features_embedding, features_edge_index, num_epochs, threshold, patience, embedding_size)

print("target :", target, "\n")

gnn_model.best_partion = torch.argmax(gnn_model.best_partion, dim=1)
print("predicted row labels :", gnn_model.best_partion,"\n")

print(f"nmi: {nmi(target, gnn_model.best_partion.cpu())}")
print(f"ari: {ari(target, gnn_model.best_partion.cpu())}")

## uncomment the lines below to plot tau functions

#fig, ax = plt.subplots()
#ax.plot(gnn_model.tau_x)
#ax.plot(gnn_model.tau_y)
#plt.plot([(gnn_model.tau_x[i] + gnn_model.tau_y[i])/2 for i in range(len(gnn_model.tau_x))])
#ax.legend(['tau x','tau y','avg tau'])
#ax.set_xlabel('iterations')
#ax.set_ylabel('tau')
#plt.show()
# load data
