from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.epochs = 5
params.slicing = [0.1]

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
