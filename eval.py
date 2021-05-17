from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 3000
params.kappa = 1500
params.epsilon = 0.15
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 15
params.pretrain = False
params.sim_length = 7
params.models = ['distilbert-base-uncased', 'roberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = True

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)

