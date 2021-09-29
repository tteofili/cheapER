from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.models = ['distilbert-base-uncased']
params.deeper_trick = False
params.sigma = 3000
params.kappa = 0
params.lr = 1e-5
params.lr_multiplier = 10
params.teaching_iterations = 5
params.epsilon = 0
params.epochs = 7
params.consistency = False
params.balance = [0.5, 0.5]
params.batch_size = 8
params.adaptive_ft = False
params.compare = False
params.silent = True
params.model_type = 'noisy-student'
params.data_noise = True

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)


