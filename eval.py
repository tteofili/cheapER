import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 5000
params.kappa = 100
params.epsilon = 0.1
params.slicing = [0.05, 0.1, 0.33, 0.5, 0.7, 1]
params.lr = 5e-5
params.epochs = 15
params.adaptive_ft = True
params.sim_length = 5
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.deeper_trick = False
params.consistency = True
params.warmup = True
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 2
params.sim_edges = False
params.adjust_ds_size = False

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)

