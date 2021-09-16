from __future__ import print_function

import logging
import operator
from inspect import getsource
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from cheaper.data.create_datasets import create_datasets
from cheaper.data.csv2dataset import csv_2_datasetALTERNATE, check_anhai_dataset, parsing_anhai_nofilter
from cheaper.data.plot import plot_dataPT
from cheaper.emt.logging_customized import setup_logging

get_lambda_name = lambda l: getsource(l).split('=')[0].strip()

setup_logging()

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
            logging.info(f'{k}:{name}:{mse}')
            if mse < lowestMSE:
                lowestMSE = mse
                bestFun = simf
                logging.info(f'update for {k}: function={get_lambda_name(bestFun)}, MSE={lowestMSE}')

        logging.info(f'BEST for {k}: function={get_lambda_name(bestFun)}, MSE={lowestMSE}')
        best.append(bestFun)
    single_sim = create_single_sim(best)
    logging.info(f'final aggregated function: {single_sim}')
    return single_sim


def create_single_sim(bf_fun):
    per_att_sim = lambda t1, t2: [np.sum(np.array([bf_fun[i](t1[i], t2[i]) for i in range(len(bf_fun))])) / len(bf_fun)]
    return per_att_sim


def learn_best_aggregate(gt_file, t1_file, t2_file, attr_indexes, sim_functions, cut, num_funcs, check=False,
                         normalize=True, lm='perceptron', deeper_trick=False):
    best = []
    i = 1
    for k in attr_indexes:
        logging.info('getting attribute values')
        if deeper_trick:
            data = check_anhai_dataset(gt_file, t1_file, t2_file, [k], sim_functions[2])
        else:
            data = parsing_anhai_nofilter(gt_file, t1_file, t2_file, [k], sim_functions[2])
        bound = int(len(data) * cut)
        data = data[:bound]

        npdata = np.array(data, dtype=object)
        X = np.zeros([len(npdata), len(sim_functions)])
        Y = np.zeros(len(npdata))
        tc = 0
        logging.info('building training set')
        ar = np.zeros(len(sim_functions))
        for t in npdata:
            arc = 0
            for s in sim_functions:
                ar[arc] = np.nan_to_num(s(t[0][0], t[1][0])[0])
                arc += 1
            X[tc] = ar
            Y[tc] = t[3]
            tc += 1
        logging.info('fitting classifier')
        score = 0
        if lm == 'perceptron':
            clf = linear_model.SGDClassifier(loss='perceptron')
        elif lm == 'ridge':
            clf = linear_model.Ridge(fit_intercept=False)
        elif lm == 'logistic':
            clf = linear_model.LogisticRegression(fit_intercept=False)
        elif lm == 'linear':
            clf = linear_model.LinearRegression(fit_intercept=False)
        r = 0
        while score < 0.9 and r < i:
            clf.fit(X, Y)
            score = clf.score(X, Y)
            r += 1
        logging.info(f'score: {score}')
        if lm == 'ridge' or lm == 'linear':
            weights = clf.coef_
        else:
            weights = clf.coef_[0]
        comb = []
        combprint = []
        normalized_weights = weights
        if normalize and min(weights) < 0:
            normalized_weights = normalized_weights + abs(min(weights))
        wsum = np.sum(normalized_weights)
        for c in range(len(weights)):
            comb.append([sim_functions[c], normalized_weights[c] / wsum])
            combprint.append([get_lambda_name(sim_functions[c]), normalized_weights[c] / wsum])
        comb.sort(key=operator.itemgetter(1), reverse=True)
        combprint.sort(key=operator.itemgetter(1), reverse=True)

        logging.info(f'sim weights for {k}: {combprint}')

        best.append(comb)

    fsims = []
    ind = 0
    for bsk in best:
        top_sims_k = bsk[:num_funcs]
        sw = np.sum(np.array(top_sims_k)[:, 1])
        for w in top_sims_k:
            w[1] = w[1] / sw
        logging.info(f'for attributes {attr_indexes[ind]}:')
        for bsa in top_sims_k:
            logging.info(f'{get_lambda_name(bsa[0])}, w:{bsa[1]}')
        fsims.append(top_sims_k)
        ind += 1

    c_data = csv_2_datasetALTERNATE(gt_file, t1_file, t2_file, attr_indexes, sim_functions[2], cut=cut)

    npdata = np.array(c_data, dtype=object)
    X = np.zeros([len(npdata), len(fsims)])
    Y = np.zeros(len(npdata))
    tc = 0
    logging.info('building agg-sim training set')
    ar = np.zeros(len(fsims))
    for t in npdata:
        arc = 0
        for a in attr_indexes:
            ar[arc] = att_sim(fsims, arc, t[0][a[0] - 1], t[1][a[1] - 1])
            arc += 1
        X[tc] = ar
        Y[tc] = t[3]
        tc += 1
    logging.info('fitting agg-sim classifier')
    score = 0
    if lm == 'perceptron':
        clf = linear_model.SGDClassifier(loss='perceptron')
    elif lm == 'ridge':
        clf = linear_model.Ridge(fit_intercept=False)
    elif lm == 'logistic':
        clf = linear_model.LogisticRegression(fit_intercept=False)
    elif lm == 'linear':
        clf = linear_model.LinearRegression(fit_intercept=False)
    r = 0
    while score < 0.9 and r < i:
        clf.fit(X, Y)
        score = clf.score(X, Y)
        r += 1
    logging.info(f'agg-sim score: {score}')
    if lm == 'ridge' or lm == 'linear':
        f_weights = clf.coef_
    else:
        f_weights = clf.coef_[0]

    if normalize and min(f_weights) < 0:
        f_weights = f_weights + abs(min(f_weights))

    f_weights = f_weights / np.sum(f_weights)
    logging.info(f_weights)

    generated_sim = lambda t1, t2: agg_sim(fsims, t1, t2, weights=f_weights)
    if check:
        final_sim = \
        find_best_simfunction(gt_file, t1_file, t2_file, attr_indexes, True, sim_functions + [generated_sim],
                              0, 100, i, 1)[0]
        return final_sim
    else:
        return generated_sim


def att_sim(best_sims, a, t1, t2):
    att_sim = 0
    for j in range(len(best_sims[a])):
        att_sim += best_sims[a][j][0](t1, t2)[0] * best_sims[a][j][1]
    return att_sim


def agg_sim(best_sims, t1, t2, weights=None):
    vect = []
    for i in range(len(t1)):
        att_sim = 0
        for j in range(len(best_sims[i])):
            att_sim += best_sims[i][j][0](t1[i], t2[i])[0] * best_sims[i][j][1]
        vect.append(att_sim)
    if weights is None:
        res = round(sum(vect) / len(vect), 2)
    else:
        res = 0
        v = 0
        for w in weights:
            if w > 0.001:
                res += vect[v] * w
            v += 1
    return [res]


def find_best_simfunction(gt_file, t1_file, t2_file, indexes, flagAnhai, simfunctions, soglia, tot_pt, tot_copy, runs):
    best = []
    for r in range(runs):
        bestFun = lambda t1, t2: t1 == t2
        lowestMSE = 1e10
        # for each sim function
        for simf in simfunctions:
            logging.info(f'using sim {get_lambda_name(simf)}')
            data, train, valid, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file, t2_file, indexes,
                                                                                     simf,
                                                                                     "sanity_check", tot_pt, flagAnhai,
                                                                                     soglia,
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
                logging.info(f'{get_lambda_name(simf)} -> mse({mse})')
                if (mse < lowestMSE):
                    lowestMSE = mse
                    bestFun = simf
                    logging.info("update: function=" + get_lambda_name(bestFun) + "', MSE=" + str(lowestMSE))

        logging.info("best function is '" + get_lambda_name(bestFun) + "' with MSE=" + str(lowestMSE))
        best.append(bestFun)
    return best
