from cheaper.params import CheapERParams
from cheaper.pipeline import cheaper_train
from cheaper.pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams(fast=True)
params.models = ['microsoft/MiniLM-L12-H384-uncased']
params.slicing = [0.05]

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
