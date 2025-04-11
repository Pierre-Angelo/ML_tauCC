import numpy as np
import torch

from tauccML_GNN import TwoGNN 

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import pandas as pd

import os 
import random

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
  #print(f"Random seed set as {seed}")

def adj_correlation(data,percentile = 90):
  adj = np.corrcoef(data)
  adj[np.isnan(adj)] = 0
  adj[adj < np.percentile(adj,percentile)] = 0
  return adj

def adj_cooccurence(data,device = "cpu", percentile = 90):
  data = torch.from_numpy(data).to(device).to(torch.float32)
  adj = torch.matmul(data,data.T).cpu().numpy()
  np.fill_diagonal(adj,0)

  adj[adj < np.percentile(adj,percentile)] = 0

  return adj

if __name__ == "__main__":

  if torch.cuda.is_available() :
    print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
    dev = "cuda:0"
  else :
    dev = "cpu"

  print(f"Device : {dev}")
  device = torch.device(dev)  

  dataset = 'tr11' # cstr, tr23, tr11, tr45, tr41, classic3, hitech, k1b, reviews, sports
  target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
  target = np.array(target_CSV).T[0]

  input_table = np.load(f'./data/{dataset}.npy')

  input_dimx, input_dimy = input_table.shape
  # Parameters
  hidden_dim = 128
  explained_variance = 0.5
  embedding_size = 10
  num_epochs = 100
  num_layers = 2
  learning_rate = 1e-3
  exp_schedule = 1
  threshold = 0.2
  patience = 20
  edge_percentile = 80
  dropout = 0
  w_decay = 0
  dtype = torch.float32
 
  set_seed(0)

  #Generate feature vector and adjacency matrix (directly conveted into edge index)

  if explained_variance == 1 :
    objects_embedding = torch.from_numpy(input_table).to(dtype).to(device)
    features_embedding = torch.from_numpy(input_table.T).to(dtype).to(device)
  else :
    objects_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_x_{explained_variance}.npy')).to(dtype).to(device)
    features_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_y_{explained_variance}.npy')).to(dtype).to(device)

  objects_adj_matrix = torch.from_numpy(adj_correlation(input_table,edge_percentile)).to_sparse_csr().to(dtype).to(device)
  features_adj_matrix = torch.from_numpy(adj_correlation(input_table.T,edge_percentile)).to_sparse_csr().to(dtype).to(device)

  data = torch.from_numpy(input_table).to(dtype).to(device)
  gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], hidden_dim, embedding_size, num_layers, learning_rate, exp_schedule, dropout,w_decay , data, device)


  print("training start") 
  gnn_model.fit(objects_embedding, objects_adj_matrix, features_embedding, features_adj_matrix, num_epochs, threshold, patience, embedding_size)

  print("target :", target, "\n")

  gnn_model.best_partion = torch.argmax(gnn_model.best_partion, dim=1)
  print("predicted row labels :", gnn_model.best_partion,"\n")

  print(f"nmi: {nmi(target, gnn_model.best_partion.cpu())}")
  print(f"ari: {ari(target, gnn_model.best_partion.cpu())}")

  ## uncomment the lines below to plot tau functions
  #import matplotlib.pyplot as plt
  #fig, ax = plt.subplots()
  #ax.plot(gnn_model.tau_x)
  #ax.plot(gnn_model.tau_y)
  #plt.plot([(gnn_model.tau_x[i] + gnn_model.tau_y[i])/2 for i in range(len(gnn_model.tau_x))])
  #ax.legend(['tau x','tau y','avg tau'])
  #ax.set_xlabel('iterations')
  #ax.set_ylabel('tau')
  #plt.show()

