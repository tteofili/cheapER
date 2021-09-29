from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.models = ['distilbert-base-uncased']
params.deeper_trick = True
params.sigma = 109
params.kappa = 10
params.lr = 1e-5
params.lr_multiplier = 0.5
params.teaching_iterations = 5
params.epsilon = 0
params.epochs = 5
params.consistency = True
params.balance = [0.5, 0.5]
params.batch_size = 4
params.adaptive_ft = False
params.compare = False
params.silent = True
params.model_type = 'noisy-student'
params.data_noise = True

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[11], params)
print(results_df)


