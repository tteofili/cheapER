from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.slicing = [0.1, 0.33]
params.lr = 5e-5
params.best_model = 'eval_loss'
params.deeper_trick = False
params.batch_size = 8

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[5], params)
print(results_df)
