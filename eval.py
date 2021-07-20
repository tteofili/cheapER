import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 1000
params.kappa = 250
params.epsilon = 0
params.slicing = [0.5]
params.adaptive_ft = True
params.sim_length = 2
params.epochs = 7
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base', 'roberta-base']
params.compare = True
params.deeper_trick = True
params.consistency = False
params.warmup = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 8
params.sim_edges = True
params.adjust_ds_size = False
params.silent = False
params.simple_slicing = False

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)

