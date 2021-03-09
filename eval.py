from pipeline import cheaper_train
from pipeline import get_datasets
import pandas as pd

datasets = get_datasets()
results = pd.DataFrame()
for d in datasets:
    results.append(cheaper_train(d, 3000, 50, 0, [0.05, 0.1, 0.15, 0.2, 0.33, 0.5, 0.67, 1], lr=1e-4, epochs=15, compare=True, sim_length=5,
                  models=['distilbert-base-uncased', 'roberta-base']))

