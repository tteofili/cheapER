from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.models = ['distilbert-base-uncased']
params.model_type = 'noisy-student'
params.adaptive_ft = False
params.epochs = 3
params.epsilon = 0.015
params.sim_length = 5
params.slicing = [0.05, 0.1, 0.33]
# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)

