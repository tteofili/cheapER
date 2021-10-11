import pandas as pd

from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5]
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base', 'roberta-base']
params.deeper_trick = True
params.sigma = 100
params.kappa = 10
params.lr = 1e-7
params.lr_multiplier = 200
params.teaching_iterations = 3
params.epsilon = 0
params.epochs = 15
params.consistency = True
params.balance = [0.5, 0.5]
params.batch_size = 8
params.adaptive_ft = False
params.compare = False
params.silent = True
params.model_type = 'noisy-student'
params.data_noise = True

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    d_res = cheaper_train(d, params)
    results.append(d_res)
    print(f'{d[4]}:\n{d_res}')
print(results)

