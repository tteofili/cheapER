from cheaper.params import CheapERParams
from cheaper.pipeline import cheaper_train
from cheaper.pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams(fast=True)
params.slicing = [0.05]
params.sigma = 10
params.kappa = 2
params.epochs = 3
params.adaptive_ft = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
