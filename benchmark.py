import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from tauccML_GNN import TwoGNN 

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.decomposition import PCA

import random
import os
from time import perf_counter
from statistics import mean, stdev

if torch.cuda.is_available() :
  print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
  dev = "cuda:0"
else :
  dev = "cpu"

print(f"Device : {dev}")
device = torch.device(dev)  

def set_seed(seed = 0,) :
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

def adj_correlation(data,percentile = 95):
  adj = np.corrcoef(data)
  adj[np.isnan(adj)] = 0
  adj[adj < np.percentile(adj,percentile)] = 0

  
  return adj

def adj_cooccurence(data, percentile = 95):
  adj = np.matmul(data,data.T)
  np.fill_diagonal(adj,0)
  adj[adj < np.percentile(adj,percentile)] = 0

  return adj

def format_plot(l):
  for res in l :
    print(f"({res[0]},{res[1]:.2f}) +- (0,{res[2]:.2f})")
  print()

ltime = []
lnmi = []
lari = []

datasets = ["cstr","tr11","classic3", "hitech", "k1b", "reviews", "sports"] # ["cstr","tr11","classic3", "hitech", "k1b", "reviews", "sports"] tr41

for dataset in datasets:
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


  objects_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_x_0.8.npy')).to(dtype).to(device)
  objects_edge_index = torch.from_numpy(adj_correlation(input_table,edge_percentile)).nonzero().t().contiguous().to(device)

  features_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_y_0.8.npy')).to(dtype).to(device)
  features_edge_index = torch.from_numpy(adj_correlation(input_table.T,edge_percentile)).nonzero().t().contiguous().to(device)

  data = torch.from_numpy(input_table).to(dtype).to(device)


  durations = []
  NMI_table = []
  ARI_table = []
  set_seed()

  print("evaluating",dataset,":")
  for i in range(5):
    gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], hidden_dim, embedding_size, num_layers, learning_rate, exp_schedule, data, device)

    start = perf_counter()
    gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], hidden_dim, embedding_size, num_layers, learning_rate, exp_schedule, data, device)
    duration = perf_counter() - start

    gnn_model.best_partion = torch.argmax(gnn_model.best_partion, dim=1)

    durations.append(duration)        
    ARI_table.append(ari(target, gnn_model.best_partion.cpu()))
    NMI_table.append(nmi(target, gnn_model.best_partion.cpu()))
  
  print(f"{dataset} : \n  Duration : mean = {mean(durations):.2f} seconds, stdev = {stdev(durations):.2f} seconds\n  NMI : mean = {mean(NMI_table):.2f}, stdev = {stdev(NMI_table):.2f}\n  ARI : mean = {mean(ARI_table):.2f}, stdev = {stdev(ARI_table):.2f}\n")

  ltime.append((dataset,mean(durations),stdev(durations)))
  lnmi.append((dataset,mean(NMI_table),stdev(NMI_table)))
  lari.append((dataset,mean(ARI_table),stdev(ARI_table)))

print("time")
format_plot(ltime)

print("nmi")
format_plot(lnmi)

print("ari")
format_plot(lari)
