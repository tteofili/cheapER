from cheaper.params import CheapERParams
from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

# cheapER parameters' settings
params = CheapERParams()
params.sigma = 300
params.kappa = 150
params.epsilon=0.05
params.slicing=[0.1, 0.33]
params.lr=5e-5
params.epochs=5
params.pretrain=False
params.sim_length = 5
params.models = ['roberta-base']
params.identity = False
params.symmetry = False
params.attribute_shuffle = False

# get datasets
datasets = get_datasets()

# perform training and collect F1 for each dataset
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, params))
print(results)
