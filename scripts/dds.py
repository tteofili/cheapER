from cheaper.params import CheapERParams
from cheaper.pipeline import cheaper_train
from cheaper.pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.lr = 2e-5

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[2], params)
print(results_df)

