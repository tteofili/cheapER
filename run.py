from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 300
params.kappa = 150
params.epsilon = 0
params.slicing = [0.05]
params.lr = 2e-5
params.epochs = 3
params.adaptive_ft = False
params.sim_length = 5
params.models = ['microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.silent = False
params.consistency = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 16
params.sim_edges = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

