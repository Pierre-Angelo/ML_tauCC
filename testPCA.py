import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from time import perf_counter

dataset = 'k1b' # cstr, tr11, classic3, hitech, k1b, reviews, sports, tr41

input_table = np.load(f'./data/{dataset}.npy')

scaler = StandardScaler()
pca = PCA(0.8)
sparsepca = SparsePCA(n_components=10,n_jobs=32) 

scaled_input = scaler.fit_transform(input_table)

start = perf_counter()
pca.fit_transform(input_table)
duration = perf_counter() - start
print("PCA",duration)


start = perf_counter()
sparsepca.fit_transform(input_table)
duration = perf_counter() - start
print("Sparse PCA",duration)