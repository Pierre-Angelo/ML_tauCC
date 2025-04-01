import numpy as np
import torch
from main_tauccGNN import set_seed, adj_correlation, adj_cooccurence
import argparse
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import pandas as pd
from tauccML_GNN import TwoGNN 


import itertools
from joblib import Parallel, delayed
import random

if torch.cuda.is_available() :
    print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
    dev = "cuda:0"
else :
    dev = "cpu"

print(f"Device : {dev}")

device = torch.device(dev) 

dataset= "cstr"
target_CSV = pd.read_csv(f'./datasets/{dataset}_target.txt', header = None)
target = np.array(target_CSV).T[0]

input_table = np.load(f'./data/{dataset}.npy')
input_dimx, input_dimy = input_table.shape
dtype = torch.float32

class CustomGridSearch:
    def __init__(self, param_grid, scoring_function, n_jobs=1):
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self):
        # Generate all possible combinations of parameters
        param_combinations = list(itertools.product(*(self.param_grid[param] for param in self.param_grid)))
        random.shuffle(param_combinations)
        print('Total combinations:', len(param_combinations))
        
        # Wrapper function to evaluate a single combination of parameters
        def evaluate_params(params):
            param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
            print(param_dict)
            score = self.scoring_function(**param_dict)
            return param_dict, score

        # Parallel processing of parameter combinations
        results = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_params)(params) for params in param_combinations)
        
        # Process results to find the best parameters and score
        for param_dict, score in results:
            self.results_.append({'params': param_dict, 'score': score})
            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = param_dict
        
        return self

    def get_results(self):
        return self.results_



# Define the search space for hyperparameters
params  = {
   'lr': [1e-3,1e-4],
    'num_layers': [2,3,4],
    'hidden_dim': [128,256,512,1024],
    'link_percent' : [80,90,98],
    'explained_var' : [0.5,0.8,0.9,1] 
}


def scoring_function(**params):

    embedding_size = 10
    num_epochs = 100
    exp_schedule = 1
    threshold = 0.2
    patience = 20

    if params['explained_var'] == 1 :
        objects_embedding = torch.from_numpy(input_table).to(dtype).to(device)
        features_embedding = torch.from_numpy(input_table.T).to(dtype).to(device)
    else :
        objects_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_x_{params["explained_var"]}.npy')).to(dtype).to(device)
        features_embedding = torch.from_numpy(np.load(f'./data/{dataset}_PCA_y_{params["explained_var"]}.npy')).to(dtype).to(device)

    objects_adj_matrix = torch.from_numpy(adj_correlation(input_table,params['link_percent'])).to_sparse_csr().to(dtype).to(device)
    features_adj_matrix = torch.from_numpy(adj_correlation(input_table.T,params['link_percent'])).to_sparse_csr().to(dtype).to(device)

    data = torch.from_numpy(input_table).to(dtype).to(device)
    gnn_model = TwoGNN(input_dimx, input_dimy, objects_embedding.shape[1], features_embedding.shape[1], params['hidden_dim'], embedding_size, params['num_layers'], params['lr'], exp_schedule, data, device)

    gnn_model.fit(objects_embedding, objects_adj_matrix, features_embedding, features_adj_matrix, num_epochs, threshold, patience, embedding_size,verbose=False)

    gnn_model.best_partion = torch.argmax(gnn_model.best_partion, dim=1)

    score = (nmi(target, gnn_model.best_partion.cpu()) + ari(target, gnn_model.best_partion.cpu())) * 0.5

    return score


if __name__ == '__main__':
    #cstr, tr23, tr11, tr45, tr41, classic3, hitech, k1b, reviews, sports
    
    n_jobs = 10

    set_seed()

    grid_search = CustomGridSearch(params, scoring_function, n_jobs=n_jobs)
    grid_search.fit()
    
    # Retrieve results
    results = grid_search.get_results()
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    #print("All Results:", results)

#{'lr': 0.0001, 'num_layers': 3, 'hidden_dim': 1024, 'link_percent': 98, 'explained_var': 0.5}