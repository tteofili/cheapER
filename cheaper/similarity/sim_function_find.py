from __future__ import print_function
from plot import plot_graph, plot_pretrain, plot_dataPT

import random
import matplotlib.pyplot as plt
from csv2dataset import csv_2_datasetALTERNATE, csvTable2datasetRANDOM_likeGold
from minhash_lsh2 import minHash_lsh
from sim_function import min_cos
from random import shuffle
import numpy as np
from create_datasets import create_datasets

from inspect import getsource

get_lambda_name = lambda l: getsource(l).strip()

def find_best_simfunction(gt_file, t1_file, t2_file, indexes, flagAnhai, simfunctions, soglia, tot_pt, tot_copy):
    best = []
    for r in range(1):
        bestFun = lambda t1, t2: t1 == t2
        lowestMSE = 1e10
        # for each sim function
        for simf in simfunctions:
            print(f'using sim {get_lambda_name(simf)}')
            data, train, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file, t2_file, indexes, simf,
                                                                              "sanity_check", tot_pt, flagAnhai, soglia,
                                                                              tot_copy, 1)
            if len(vinsim_data) > 0:
                shuffle(vinsim_data)

                plt.xlabel(get_lambda_name(simf))

                t, sim_list = plot_dataPT(vinsim_data)
                plt.xlabel('')
                gradino = []
                for g in range(len(t)):
                    if g >= len(t) / 2:
                        gradino.append(1)
                    else:
                        gradino.append(0)
                mse = (np.square(np.array(gradino) - sim_list)).mean(axis=None)
                print(f'{get_lambda_name(simf)} -> mse({mse})')
                if (mse < lowestMSE):
                    lowestMSE = mse
                    bestFun = simf
                    print("update: function="+get_lambda_name(bestFun)+"', MSE="+str(lowestMSE))

        print("best function is '"+get_lambda_name(bestFun)+"' with MSE="+str(lowestMSE))
        best.append(bestFun)
    return best