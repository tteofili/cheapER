from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.epochs = 3
params.adaptive_ft = False
params.threshold = 0.5

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[9], params)
print(results_df)
