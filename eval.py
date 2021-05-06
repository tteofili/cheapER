from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

datasets = get_datasets()
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, 300, 150, 0, [0.1, 0.33, 0.5, 0.7, 1], lr=1e-5, epochs=7, compare=True, sim_length=9,
                  models=['microsoft/deberta-base'], pretrain=True, attribute_shuffle=True, identity=True, symmetry=True))
print(results)
