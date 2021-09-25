from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.models = ['distilbert-base-uncased']
params.model_type = 'noisy-student'
params.adaptive_ft = False
params.epochs = 5
params.deep_trick = True
params.epsilon = 0.0
params.kappa = 0
params.sigma = 1000
params.epsilon = 0
params.sim_length = 5
params.consistency = False
params.warmup = False
params.batch_size = 8
params.compare = False
params.silent = True
params.slicing = [0.05, 0.1, 0.33]
# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)


