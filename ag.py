from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets

# cheapER parameters' settings
params = CheapERParams()
params.lr = 1e-6
params.lr_multiplier = 10
params.sigma = 100
params.kappa = 10
params.adjust_ds_size = False
params.consistency = True
params.temperature = None
params.discard_old_data = False
params.threshold = 0
params.epsilon = 0
params.label_smoothing = 0.2
params.slicing = [0.05]
params.epochs = 5

# get datasets
datasets = get_datasets()
results_df = cheaper_train(datasets[6], params)
print(results_df)

