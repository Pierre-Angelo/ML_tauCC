import numpy as np
import torch

from tauccML_GNN import TwoGNN 
from main_tauccGNN import set_seed, adj_correlation, adj_cooccurence

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari

import pandas as pd

from time import perf_counter
from statistics import mean, stdev

if torch.cuda.is_available() :
  print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
  dev = "cuda:0"
else :
  dev = "cpu"

print(f"Device : {dev}")
device = torch.device(dev)  

def format_plot(l):
  for res in l :
    print(f"({res[0]},{res[1]:.2f}) +- (0,{res[2]:.2f})")
  print()

ltime = []
lnmi = []
lari = []

# ["cstr","tr11","classic3", "hitech", "k1b", "reviews", "sports"] tr41
datasets = ["cstr","tr11","classic3", "hitech", "k1b", "reviews", "sports"] 

for dataset in datasets:
  target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
  target = np.array(target_CSV).T[0]

  input_table = np.load(f'./data/{dataset}.npy')

  input_dimx, input_dimy = input_table.shape

  # Parameters
  hidden_dim = 128
  explained_variance = 0.8
  embedding_size = 10
  num_epochs = 50
  num_layers = 2
  learning_rate = 1e-3
  exp_schedule = 1
  threshold = 0.05
  patience = 10
  edge_percentile = 99 
  dtype = torch.float32


  objects_embedding = torch.from_numpy(input_table).to(dtype).to(device)
  objects_edge_index = torch.from_numpy(adj_correlation(input_table)).nonzero().t().contiguous().to(device)

  features_embedding = torch.from_numpy(input_table.T).to(dtype).to(device)
  features_edge_index = torch.from_numpy(adj_correlation(input_table.T)).nonzero().t().contiguous().to(device)

  data = torch.from_numpy(input_table).to(dtype).to(device)


  durations = []
  NMI_table = []
  ARI_table = []
  set_seed()

  print("evaluating",dataset,":")
  for i in range(5):
    gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], hidden_dim, embedding_size, num_layers, learning_rate, exp_schedule, data, device)

    start = perf_counter()
    gnn_model.fit(objects_embedding, objects_edge_index, features_embedding, features_edge_index, num_epochs, threshold, patience, embedding_size,verbose = False)
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
