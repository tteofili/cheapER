from pipeline import cheaper_train
from pipeline import get_datasets

datasets = get_datasets()
for d in datasets:
    cheaper_train(d, 5000, 50, 0, [0.05, 0.15, 0.33, 0.5, 1], lr=1e-4, epochs=4, compare=True, sim_length=2)
