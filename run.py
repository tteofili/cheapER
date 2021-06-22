from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 100
params.kappa = 50
params.epsilon = 0
params.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 5
params.pretrain = False
params.sim_length = 15
params.models = ['distilbert-base-uncased']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

