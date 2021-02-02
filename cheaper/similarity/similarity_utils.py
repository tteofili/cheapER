from __future__ import print_function

import operator
from inspect import getsource
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from cheaper.data.create_datasets import create_datasets
from cheaper.data.csv2dataset import csv_2_datasetALTERNATE
from cheaper.data.plot import plot_dataPT

get_lambda_name = lambda l: getsource(l).split('=')[0].strip()


def unflat(data):
    u_data = np.zeros_like(data)
    i = 0
    for d in data:
        u_data[i] = d[0]
        i += 1
    return u_data


def brute_force_per_attribute(gt_file, t1_file, t2_file, attr_indexes, sim_functions):
    best = []
    for k in attr_indexes:
        bestFun = lambda t1, t2: t1 == t2
        lowestMSE = 1e10
        for simf in sim_functions:
            name = get_lambda_name(simf)
            data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, [k], sim_functions[2])
            perc = len(data) * 0.05
            split = int(min(perc / 2, 50))
            npdata = np.array(data[:split] + data[-split:])
            npdata[:, 2] = unflat(npdata[:, 2])
            ones = npdata[np.where(npdata[:, 3] == 1)][:, 3]
            ones_sim = npdata[np.where(npdata[:, 3] == 1)][:, 2]
            zeros = npdata[np.where(npdata[:, 3] == 0)][:, 3]
            zeros_sim = npdata[np.where(npdata[:, 3] == 0)][:, 2]
            mse_ones = np.square((ones - ones_sim)).mean(axis=None)
            mse_zeros = 0
            if len(zeros_sim > 0):
                mse_zeros = np.square((zeros - zeros_sim)).mean(axis=None)
            alpha = 0.5 + (len(ones_sim) - len(zeros_sim)) / (len(ones_sim) + len(zeros_sim))
            mse = mse_ones * alpha + mse_zeros * (1 - alpha)
            print(f'{k}:{name}:{mse}')
            if mse < lowestMSE:
                lowestMSE = mse
                bestFun = simf
                print(f'update for {k}: function={get_lambda_name(bestFun)}, MSE={lowestMSE}')

        print(f'BEST for {k}: function={get_lambda_name(bestFun)}, MSE={lowestMSE}')
        best.append(bestFun)
    single_sim = create_single_sim(best)
    print(f'final aggregated function: {single_sim}')
    return single_sim


def create_single_sim(bf_fun):
    per_att_sim = lambda t1, t2: [np.sum(np.array([bf_fun[i](t1[i], t2[i]) for i in range(len(bf_fun))])) / len(bf_fun)]
    return per_att_sim


def learn_best_aggregate(gt_file, t1_file, t2_file, attr_indexes, sim_functions, cut, num_funcs, check=False):
    best = []
    for k in attr_indexes:
        print('getting attribute values')
        data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, [k], sim_functions[2], cut=cut)
        npdata = np.array(data, dtype=object)
        X = np.zeros([len(npdata), len(sim_functions)])
        Y = np.zeros(len(npdata))
        tc = 0
        print('building training set')
        for t in npdata:
            ar = np.zeros(len(sim_functions))
            arc = 0
            for s in sim_functions:
                ar[arc] = np.nan_to_num(s(t[0][0], t[1][0])[0])
                arc += 1
            X[tc] = ar
            Y[tc] = t[3]
            tc += 1
        print('fitting classifier')
        score = 0
        clf = linear_model.SGDClassifier(loss='perceptron')
        r = 0
        while (score < 0.9 and r < 50):
            clf.fit(X, Y)
            score = clf.score(X, Y)
            r += 1
        print(f'score: {score}')
        weights = clf.coef_[0]
        comb = []
        combprint = []
        normalized_weights = weights
        if min(weights) < 0:
            normalized_weights = normalized_weights + abs(min(weights))
        wsum = np.sum(normalized_weights)
        for c in range(len(weights)):
            comb.append([sim_functions[c], normalized_weights[c] / wsum])
            combprint.append([get_lambda_name(sim_functions[c]), normalized_weights[c] / wsum])
        comb.sort(key=operator.itemgetter(1), reverse=True)
        combprint.sort(key=operator.itemgetter(1), reverse=True)

        print(f'sim weights for {k}: {combprint}')

        best.append(comb)

    fsims = []
    ind = 0
    for bsk in best:
        top_sims_k = bsk[:num_funcs]
        sw = np.sum(np.array(top_sims_k)[:, 1])
        for w in top_sims_k:
            w[1] = w[1] / sw
        print(f'for attributes {attr_indexes[ind]}:')
        for bsa in top_sims_k:
            print(f'{get_lambda_name(bsa[0])}, w:{bsa[1]}')
        fsims.append(top_sims_k)
        ind += 1
    generated_sim = lambda t1, t2: agg_sim(fsims, t1, t2)
    if check:
        final_sim = \
        find_best_simfunction(gt_file, t1_file, t2_file, attr_indexes, True, sim_functions + [generated_sim], 0,
                              100, 50, 1)[0]
        return final_sim
    else:
        return generated_sim


def agg_sim(best_sims, t1, t2):
    vect = []
    for i in range(len(t1)):
        att_sim = 0
        for j in range(len(best_sims[i])):
            att_sim += best_sims[i][j][0](t1[i], t2[i])[0] * best_sims[i][j][1]
        vect.append(att_sim)
    aver = round(sum(vect) / len(vect), 2)
    return [aver]


def find_best_simfunction(gt_file, t1_file, t2_file, indexes, flagAnhai, simfunctions, soglia, tot_pt, tot_copy, runs):
    best = []
    for r in range(runs):
        bestFun = lambda t1, t2: t1 == t2
        lowestMSE = 1e10
        # for each sim function
        for simf in simfunctions:
            print(f'using sim {get_lambda_name(simf)}')
            data, train, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file, t2_file, indexes, simf,
                                                                              "sanity_check", tot_pt, flagAnhai, soglia,
                                                                              tot_copy, 1, 1, gt_file, gt_file)
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
                    print("update: function=" + get_lambda_name(bestFun) + "', MSE=" + str(lowestMSE))

        print("best function is '" + get_lambda_name(bestFun) + "' with MSE=" + str(lowestMSE))
        best.append(bestFun)
    return best
