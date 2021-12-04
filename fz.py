from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.epochs = 7
params.slicing = [0.33, 0.5]
params.adaptive_ft = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[9], params)
print(results_df)
