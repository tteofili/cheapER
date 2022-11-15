from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.models = ['roberta-base']
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5]
params.compare = True
params.lr = 2e-5
params.mask_token = '<mask>'

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[2], params)
print(results_df)

