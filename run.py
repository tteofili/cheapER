from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.epochs = 15
params.adaptive_ft = False
params.models = ['distilroberta-base']
params.threshold = 0
params.lr = 1e-5
params.lr_multiplier = 2
params.epsilon = 0.015
params.deeper_trick = True
params.discard_old_data = True

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
