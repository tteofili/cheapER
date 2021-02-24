from __future__ import print_function

import pandas as pd
import logging
from cheaper.emt.logging_customized import setup_logging
from cheaper.data.create_datasets import create_datasets
import os
from cheaper.data.csv2dataset import splitting_dataSet
from cheaper.emt import config

from cheaper.emt.emtmodel import EMTERModel
from cheaper.similarity import sim_function
from cheaper.similarity.similarity_utils import learn_best_aggregate
from inspect import getsource
from datetime import date

simfunctions = [
    lambda t1, t2: sim_function.jaro(t1, t2),
    lambda t1, t2: sim_function.sim_lev(t1, t2),
    lambda t1, t2: sim_function.sim_cos(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1, t2),
    lambda t1, t2: sim_function.sim_hamming(t1, t2),
    lambda t1, t2: sim_function.sim_jacc(t1, t2),
    lambda t1, t2: sim_function.sim_ngram(t1, t2),
    lambda t1, t2: sim_function.sim_sodi(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_jacc(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_cos(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_hamming(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_ngram(t1.split(), t2.split()),
    lambda t1, t2: sim_function.jaro(t1.split(), t2.split()),
    lambda t1, t2: sim_function.sim_lev(t1.split(), t2.split()),
]

get_lambda_name = lambda l: getsource(l).strip()

setup_logging()

def train_model(gt_file, t1_file, t2_file, indexes, tot_pt, soglia, tot_copy, dataset_name, flag_Anhai, num_run,
                slicing, compare=False, sim_length=len(simfunctions)):
    results = pd.DataFrame()
    for n in range(num_run):
        for cut in slicing:
            simf = learn_best_aggregate(gt_file, t1_file, t2_file, indexes, simfunctions, cut, sim_length,
                                        normalize=True)

            logging.info('Generating dataset')
            # create datasets
            test_file = base_dir + dataset_name + os.sep + 'test.csv'
            valid_file = base_dir + dataset_name + os.sep + 'valid.csv'
            data, train, valid, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file,
                                                                                     t2_file, indexes, simf,
                                                                                     dataset_name,
                                                                                     tot_pt,
                                                                                     flag_Anhai, soglia, tot_copy,
                                                                                     num_run, cut, valid_file,
                                                                                     test_file)
            logging.info('Generated dataset size: {}'.format(len(vinsim_data_app)))

            train_cut = train#splitting_dataSet(cut, train)

            for model_type in config.Config.MODEL_CLASSES:

                if compare:
                    logging.info("------------- Vanilla EMT Training {model_type} ------------------")
                    logging.info('Training with {} record pairs ({}% GT)'.format(len(train_cut), 100 * cut))
                    model = EMTERModel(model_type)

                    classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = model \
                        .train(train_cut, valid, test, dataset_name, seq_length=seq_length)
                    classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = model \
                        .eval(test, dataset_name, seq_length=seq_length)
                    new_row = {'model_type': model_type, 'train': 'cl', 'cut': cut, 'pM': classic_precision, 'rM': classic_recall,
                               'f1M': classic_f1,
                               'pNM': classic_precisionNOMATCH, 'rNM': classic_recallNOMATCH, 'f1NM': classic_f1NOMATCH}
                    results = results.append(new_row, ignore_index=True)

                logging.info("------------- Data augmented EMT Training {} -----------------".format(model_type))
                dataDa = vinsim_data_app + train_cut
                logging.info('Training with {} record pairs (generated dataset {} + {}% GT)'.format(len(dataDa), len(vinsim_data_app), 100 * cut))
                model = EMTERModel(model_type)
                da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.train(
                    dataDa, valid, test, dataset_name, seq_length=seq_length)
                da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(test, dataset_name, seq_length=seq_length)
                new_row = {'model_type': model_type, 'train': 'da', 'cut': cut, 'pM': da_precision, 'rM': da_recall, 'f1M': da_f1,
                           'pNM': da_precisionNOMATCH,
                           'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                results = results.append(new_row, ignore_index=True)

                logging.info(results.to_string)

        today = date.today()
        results.to_csv(
            'results' + os.sep + today.strftime("%b-%d-%Y") + '_' + dataset_name + '_' + str(tot_pt) + '_'
            + str(tot_copy) + '_' + str(soglia) + '.csv')


# main program execution


base_dir = 'datasets' + os.sep
datasets = [
    [('%sdirty_walmart_amazon/train.csv' % base_dir), ('%sdirty_walmart_amazon/tableA.csv' % base_dir),
     ('%sdirty_walmart_amazon/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], 'dirty_walmart_amazon',
     ('%stemporary/' % base_dir), True, 150],
    [('%sdirty_amazon_itunes/train.csv' % base_dir), ('%sdirty_amazon_itunes/tableA.csv' % base_dir),
     ('%sdirty_amazon_itunes/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_amazon_itunes',
     ('%stemporary/' % base_dir), True, 180],
    [('%sdirty_dblp_scholar/train.csv' % base_dir), ('%sdirty_dblp_scholar/tableA.csv' % base_dir),
     ('%sdirty_dblp_scholar/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_dblp_scholar',
     ('%stemporary/' % base_dir), True, 128],
    [('%sdirty_dblp_acm/train.csv' % base_dir), ('%sdirty_dblp_acm/tableA.csv' % base_dir),
     ('%sdirty_dblp_acm/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dirty_dblp_acm',
     ('%stemporary/' % base_dir), True, 180],
    [('%sabt_buy/train.csv' % base_dir), ('%sabt_buy/tableA.csv' % base_dir),
     ('%sabt_buy/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3)], 'abt_buy',
     ('%stemporary/' % base_dir), True, 265],
    [('%sbeers/train.csv' % base_dir), ('%sbeers/tableA.csv' % base_dir),
     ('%sbeers/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'beers',
     ('%stemporary/' % base_dir), True, 150],
    [('%samazon_google/train.csv' % base_dir),
     ('%samazon_google/tableA.csv' % base_dir),
     ('%samazon_google/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3)], 'amazon_google',
     ('%stemporary/' % base_dir), True, 180],
    [('%sdblp_scholar/train.csv' % base_dir), ('%sdblp_scholar/tableA.csv' % base_dir),
     ('%sdblp_scholar/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dblp_scholar',
     ('%stemporary/' % base_dir), True, 180],
    [('%swalmart_amazon/train.csv' % base_dir), ('%swalmart_amazon/talbleA.csv' % base_dir),
     ('%swalmart_amazon/tableB.csv' % base_dir), [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)], 'walmart_amazon',
     ('%stemporary/' % base_dir), True, 150],
    [('%sfodo_zaga/train.csv' % base_dir), ('%sfodo_zaga/tableA.csv' % base_dir),
     ('%sfodo_zaga/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'fodo_zaga',
     ('%stemporary/' % base_dir), True, 150],
    [('%sdblp_acm/train.csv' % base_dir), ('%sdblp_acm/tableA.csv' % base_dir),
     ('%sdblp_acm/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4)], 'dblp_acm',
     ('%stemporary/' % base_dir), True, 180],
    [('%situnes_amazon/train.csv' % base_dir), ('%situnes_amazon/tableA.csv' % base_dir),
     ('%situnes_amazon/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
     'itunes_amazon',
     ('%stemporary/' % base_dir), True, 180],
]

ablation = False
train = True

if train:
    for d in datasets:
        gt_file = d[0]
        t1_file = d[1]
        t2_file = d[2]
        indexes = d[3]
        dataset_name = d[4]
        datadir = d[5]
        flag_Anhai = d[6]
        seq_length = d[7]
        logging.info('---{}---'.format(dataset_name))
        sigma = 5000  # generated dataset size
        kappa = 10  # no. of samples for consistency training
        epsilon = 0.15  # deviation from calculated min/max thresholds
        slicing = [0.01, 0.05, 0.1, 0.15, 0.33, 0.5, 0.67, 0.75, 1]
        num_runs = 1
        train_model(gt_file, t1_file, t2_file, indexes, sigma, epsilon, kappa, dataset_name, flag_Anhai, num_runs, slicing,
                    compare=False, sim_length=5)
if ablation:
    for d in datasets[:2]:
        gt_file = d[0]
        t1_file = d[1]
        t2_file = d[2]
        indexes = d[3]
        dataset_name = d[4]
        datadir = d[5]
        flag_Anhai = d[6]
        logging.info('ablation---{}---', dataset_name)
        for sigma in [100]:
            for epsilon in [0, 0.2]:
                for kappa in [0, 50]:
                    try:
                        train_model(gt_file, t1_file, t2_file, indexes, sigma, epsilon, kappa,
                                    dataset_name, flag_Anhai, 1, [1], compare=False)
                    except:
                        pass
