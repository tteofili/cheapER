from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 1000
params.kappa = 500
params.epsilon = 0
params.slicing = [0.33]
params.lr = 2e-5
params.epochs = 7
params.adaptive_ft = False
params.sim_length = 2
params.models = ['microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.silent = False
params.deeper_trick = False
params.consistency = False
params.warmup = False
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 4
params.sim_edges = False
params.adjust_ds_size = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[1], params)
print(results_df)

