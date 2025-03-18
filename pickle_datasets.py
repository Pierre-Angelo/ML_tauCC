import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

datasets = ["cstr", "tr11", "classic3", "hitech", "k1b", "reviews", "sports"]
explained_variance = 0.5
pca = PCA(explained_variance)
scaler = StandardScaler()

for dataset in datasets:
    input_CSV = pd.read_csv(f'./datasets/{dataset}.txt')

    table_size_x = len(input_CSV.doc.unique())
    table_size_y = len(input_CSV.word.unique())
    input_table = np.zeros((table_size_x,table_size_y), dtype = int)

    for row in input_CSV.iterrows():
        input_table[row[1].doc,row[1].word] = row[1].cluster

    #np.save(f"data/{dataset}",input_table)

    np.save(f"data/{dataset}_PCA_x_{explained_variance}",pca.fit_transform(scaler.fit_transform(input_table)))
    np.save(f"data/{dataset}_PCA_y_{explained_variance}",pca.fit_transform(scaler.fit_transform(input_table.T)))
