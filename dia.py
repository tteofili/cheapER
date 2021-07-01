from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 3000
params.kappa = 1500
params.epsilon = 0.15
params.slicing = [0.1, 0.33, 0.5, 0.7, 1]
params.lr = 5e-5
params.epochs = 5
params.adaptive_ft = False
params.sim_length = 5
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.consistency = False
params.warmup = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[1], params)
print(results_df)

