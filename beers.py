from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.slicing = [0.05, 0.1, 0.33]
params.batch_size = 16
params.best_model = 'eval_loss'
params.deeper_trick = False

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
