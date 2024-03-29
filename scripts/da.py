from cheaper.params import CheapERParams
from cheaper.pipeline import cheaper_train
from cheaper.pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[10], params)
print(results_df)

