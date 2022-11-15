import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams(fast=False)

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)
