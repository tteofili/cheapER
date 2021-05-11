from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

datasets = get_datasets()
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, 3000, 1500, 0.05, [0.1, 0.33, 0.5, 0.7, 1], lr=5e-5, epochs=7, compare=True, sim_length=9,
                  models=['roberta-base'], pretrain=True, attribute_shuffle=False, identity=False, symmetry=False))
print(results)
