from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 1000
params.kappa = 500
params.epsilon = 0
params.slicing = [0.33]
params.lr = 5e-5
params.epochs = 3
params.pretrain = False
params.sim_length = 2
params.models = ['distilber-base-uncased']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False
params.compare = False

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, params))
print(results)
