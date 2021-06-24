from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 100
params.kappa = 10
params.epsilon = 0.01
params.slicing = [0.05]
params.lr = 2e-5
params.epochs = 3
params.adaptive_ft = False
params.sim_length = 15
params.models = ['distilbert-base-uncased']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.silent = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 8

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[1], params)
print(results_df)

