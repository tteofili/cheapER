from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 1000
params.kappa = 100
params.epsilon = 0.15
params.slicing = [0.05, 0.1, 0.33, 0.5, 0.7, 1]
params.lr = 2e-5
params.epochs = 15
params.adaptive_ft = True
params.sim_length = 2
params.models = ['distilbert-base-uncased', 'microsoft/deberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.deeper_trick = False
params.consistency = True
params.warmup = True
params.approx = 'perceptron'
params.balance = [0.5, 0.5]
params.batch_size = 16
params.sim_edges = True
params.adjust_ds_size = False
params.silent = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[8], params)
print(results_df)

