from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.epochs = 3
params.adaptive_ft = False
params.models = ['distilbert-base-uncased']
params.lr = 2e-5
params.lr_multiplier = 1

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[0], params)
print(results_df)
