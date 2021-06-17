from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 10
params.kappa = 0
params.epsilon = 0
params.slicing = [0.1]
params.lr = 2e-5
params.epochs = 3
params.pretrain = True
params.sim_length = 7
params.models = ['distilbert-base-uncased']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False
params.generated_only = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[9], params)
print(results_df)

