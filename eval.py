import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 300
params.kappa = 150
params.epsilon = 0
params.slicing = [0.1, 0.33, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 15
params.pretrain = True
params.sim_length = 10
params.models = ['distilbert-base-uncased', 'roberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.generated_only = True
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

