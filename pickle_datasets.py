import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

datasets = ["cstr", "tr23", "tr11", "tr45", "tr41", "classic3", "hitech", "k1b", "reviews", "sports"]
datasets = ["ohscal", "ng20","wiki"]
explained_variance = 0.5
pca = PCA(explained_variance)
scaler = StandardScaler()

for dataset in datasets:
    input_CSV = pd.read_csv(f'./datasets/{dataset}.txt')

    table_size_x = max(input_CSV.doc.unique()) + 1
    table_size_y = len(input_CSV.word.unique())
    input_table = np.zeros((table_size_x,table_size_y), dtype = float)

    for row in input_CSV.iterrows():
        input_table[int(row[1].doc),int(row[1].word)] = row[1].cluster

    #input_table = np.load(f'./data/{dataset}.npy')
    
    np.save(f"data/{dataset}",input_table)
    print(dataset,input_table.shape, "sparsity :",1 - np.count_nonzero(input_table)/(table_size_x * table_size_y))
   

    np.save(f"data/{dataset}_PCA_x_{explained_variance}",pca.fit_transform(scaler.fit_transform(input_table)))
    np.save(f"data/{dataset}_PCA_y_{explained_variance}",pca.fit_transform(scaler.fit_transform(input_table.T)))

""" acm (3025, 1870) sparsity : 0.955221637866266 5 M
pubmed (19717, 500) sparsity : 0.8997787695896942 9 M
ohscal (11162, 11465) sparsity : 0.9947303842947924 127 M
ng20 (18846, 14390) sparsity : 0.9940732598965891
wiki (2405, 4973) sparsity : 0.8698506237215267 """