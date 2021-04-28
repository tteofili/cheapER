from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

datasets = get_datasets()
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, 3000, 1500, 0, [0.1, 0.2, 0.33], lr=2e-5, epochs=5, compare=False, sim_length=5,
                  models=['microsoft/deberta-base'], pretrain=True, attribute_shuffle=True))

