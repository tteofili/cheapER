from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 3000
params.kappa = 1500
params.epsilon = 0.15
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 15
params.pretrain = True
params.sim_length = 5
params.models = ['distilbert-base-uncased', 'roberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[0], params)
print(results_df)

