import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 3000
params.kappa = 750
params.epsilon = 0
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.7, 1]
params.adaptive_ft = False
params.sim_length = 5
params.epochs = 15
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base', 'roberta-base']
params.compare = True
params.deeper_trick = True
params.warmup = False
params.batch_size = 8
params.sim_edges = True
params.silent = False
params.simple_slicing = True

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)

