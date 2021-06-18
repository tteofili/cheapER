from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 100
params.kappa = 50
params.epsilon = 0
params.slicing = [0.2, 0.33, 0.4, 0.5]
params.lr = 1e-4
params.epochs = 7
params.pretrain = False
params.sim_length = 10
params.models = ['distilbert-base-uncased']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.generated_only = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

