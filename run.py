from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 100
params.kappa = 50
params.epsilon = 0
params.slicing = [0.05]
params.lr = 2e-5
params.epochs = 3
params.adaptive_ft = False
params.sim_length = 10
params.models = ['microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.silent = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 4

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

