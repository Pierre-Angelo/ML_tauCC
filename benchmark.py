import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from tauccML_GNN import TwoGNN 

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
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

def adj_corrrelation(data, edge_thr = 0):
  adj = np.corrcoef(data)
  adj[np.isnan(adj)] = 0
  adj[adj < edge_thr] = 0

  return adj

def adj_cooccurence(data, edge_thr = 1):
  adj = np.matmul(data,data.T)
  np.fill_diagonal(adj,0)
  adj[adj < edge_thr] = 0

  return adj

def format_plot(l):
  for res in l :
    print(f"({res[0]},{res[1]:.2f}) +- (0,{res[2]:.2f})")
  print()

ltime = []
lnmi = []
lari = []

datasets = ["cstr","tr11"] # cstr, tr11, classic3, hitech, k1b, reviews, sports, tr41

for dataset in datasets:
  input_CSV = pd.read_csv(f'./datasets/{dataset}.txt')
  target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
  target = np.array(target_CSV).T[0]

  table_size_x = len(input_CSV.doc.unique())
  table_size_y = len(input_CSV.word.unique())
  input_table = np.zeros((table_size_x,table_size_y), dtype = int)

  for row in input_CSV.iterrows():
    input_table[row[1].doc,row[1].word] = row[1].cluster

  # set some parameters
  hidden_size = 128
  embedding_size = 10
  num_epochs = 50
  input_dimx = table_size_x
  input_dimy = table_size_y
  hidden_dim = hidden_size
  output_dim = embedding_size
  num_layers = 2
  learning_rate = 1e-3
  exp_schedule = 0.9
  threshold = 0.05
  patience = 150
  edge_thr = 0.5
  dtype = torch.float32
  
  data = torch.from_numpy(input_table).to(dtype).to(device)
  x = data 
  edge_index_x = torch.from_numpy(adj_corrrelation(input_table,edge_thr)).nonzero().t().contiguous().to(device) 

  y = data.T 
  edge_index_y = torch.from_numpy(adj_corrrelation(input_table.T,edge_thr)).nonzero().t().contiguous().to(device)

  durations = []
  NMI_table = []
  ARI_table = []

  set_seed()

  for i in range(5):
    gnn_model = TwoGNN(input_dimx, input_dimy, hidden_dim, output_dim, num_layers, learning_rate, exp_schedule, data, device)

    start = perf_counter()
    gnn_model.fit(x, edge_index_x, y, edge_index_y, num_epochs, threshold, patience, embedding_size,verbose=False)
    duration = perf_counter() - start

    gnn_model.row_labels_ = torch.argmax(gnn_model.row_labels_, dim=1)

    durations.append(duration)        
    ARI_table.append(ari(target, gnn_model.row_labels_.cpu()))
    NMI_table.append(nmi(target, gnn_model.row_labels_.cpu()))
  
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
