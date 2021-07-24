from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 1000
params.kappa = 250
params.epsilon = 0.015
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 3
params.adaptive_ft = False
params.sim_length = 5
params.models = ['distilbert-base-uncased']

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

