from __future__ import print_function

import logging
import os
from datetime import date
from inspect import getsource

import pandas as pd

from cheaper.data.create_datasets import add_shuffle
from cheaper.data.create_datasets import create_datasets, add_identity, add_symmetry
from cheaper.data.csv2dataset import splitting_dataSet
from cheaper.emt.emtmodel import EMTERModel
from cheaper.emt.logging_customized import setup_logging
from cheaper.params import CheapERParams
from cheaper.similarity import sim_function
from cheaper.similarity.similarity_utils import learn_best_aggregate

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


def train_model(gt_file, t1_file, t2_file, indexes, dataset_name, flag_Anhai, seq_length, params: CheapERParams):
    results = pd.DataFrame()

    tableA = pd.read_csv(t1_file)
    tableB = pd.read_csv(t2_file)

    basedir = 'models/' + dataset_name
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    unlabelled_train = basedir + '/unlabelled_train.txt'
    unlabelled_valid = basedir + '/unlabelled_valid.txt'

    for n in range(params.num_runs):
        for cut in params.slicing:
            simf = learn_best_aggregate(gt_file, t1_file, t2_file, indexes, simfunctions, cut, params.sim_length,
                                        normalize=params.normalize, lm=params.approx)

            logging.info('Generating dataset')
            # create datasets
            test_file = base_dir + dataset_name + os.sep + 'test.csv'
            valid_file = base_dir + dataset_name + os.sep + 'valid.csv'
            data, train, valid, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file,
                                                                                     t2_file, indexes, simf,
                                                                                     dataset_name,
                                                                                     params.sigma,
                                                                                     flag_Anhai, params.epsilon,
                                                                                     params.kappa,
                                                                                     params.num_runs, cut, valid_file,
                                                                                     test_file, params.balance,
                                                                                     params.adjust_ds_size)
            logging.info('Generated dataset size: {}'.format(len(vinsim_data_app)))

            generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB, vinsim_data_app)

            train_cut = splitting_dataSet(cut, train)

            for model_type in params.models:

                if params.compare:
                    logging.info("------------- Vanilla EMT Training {} ------------------".format(model_type))
                    logging.info('Training with {} record pairs ({}% GT)'.format(len(train_cut), 100 * cut))
                    model = EMTERModel(model_type)

                    model.train(train_cut, valid, test, model_type, seq_length=seq_length, warmup=params.warmup,
                                epochs=params.epochs, lr=params.lr, batch_size=params.batch_size)
                    classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = model \
                        .eval(test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                    new_row = {'model_type': model_type, 'train': 'cl', 'cut': cut, 'pM': classic_precision,
                               'rM': classic_recall,
                               'f1M': classic_f1,
                               'pNM': classic_precisionNOMATCH, 'rNM': classic_recallNOMATCH, 'f1NM': classic_f1NOMATCH}
                    results = results.append(new_row, ignore_index=True)

                logging.info("------------- Data augmented EMT Training {} -----------------".format(model_type))

                dataDa = vinsim_data_app

                if params.identity:
                    dataDa = add_identity(dataDa)

                if params.symmetry:
                    dataDa = add_symmetry(dataDa)

                if params.attribute_shuffle:
                    dataDa = add_shuffle(dataDa)

                model = EMTERModel(model_type)

                if params.pretrain:
                    model.pretrain(unlabelled_train, unlabelled_valid, dataset_name, model_type, seq_length=seq_length,
                                   epochs=params.epochs, lr=params.lr)

                # generated data train only
                if params.generated_only:
                    model.train(dataDa, valid, model_type, dataset_name, seq_length=seq_length, warmup=params.warmup,
                            epochs=params.epochs, lr=params.lr, pretrain=params.pretrain, silent=params.silent,
                                batch_size=params.batch_size)


                    da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(
                        test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                    new_row = {'model_type': model_type, 'train': 'da-only', 'cut': cut, 'pM': da_precision,
                               'rM': da_recall,
                               'f1M': da_f1,
                               'pNM': da_precisionNOMATCH,
                               'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                    results = results.append(new_row, ignore_index=True)

                # gt+generated data train
                model = EMTERModel(model_type)
                model.train(train_cut + dataDa, valid, model_type, dataset_name, seq_length=seq_length, warmup=params.warmup,
                            epochs=params.epochs, lr=params.lr, pretrain=params.pretrain, silent=params.silent,
                            batch_size=params.batch_size)

                da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(test,
                                                                                                                 dataset_name,
                                                                                                                 seq_length=seq_length,
                                                                                                                 batch_size=params.batch_size)
                new_row = {'model_type': model_type, 'train': 'da', 'cut': cut, 'pM': da_precision, 'rM': da_recall,
                           'f1M': da_f1,
                           'pNM': da_precisionNOMATCH,
                           'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                results = results.append(new_row, ignore_index=True)

                logging.info(results.to_string)

        today = date.today()
        filename = 'results' + os.sep + today.strftime("%b-%d-%Y") + '_' + dataset_name + '.csv'
        with open(filename, 'a') as f:
            f.write('# ' + str(params) + '\n')
        results.to_csv(filename, mode='a')
    return results


def generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB, vinsim_data_app):
    if os.path.exists(unlabelled_train):
        os.remove(unlabelled_train)
    if os.path.exists(unlabelled_valid):
        os.remove(unlabelled_valid)

    open(unlabelled_train, 'w').close()
    open(unlabelled_valid, 'w').close()

    lines = []
    for l in tableA.values:
        row = ''
        for a in l[1:]:
            row += str(a) + ' '
        lines.append(row)
    for l in tableB.values:
        row = ''
        for a in l[1:]:
            row += str(a) + ' '
        lines.append(row)

    '''ug = []
    for pair in vinsim_data_app:
        ug.append(pair[0])
        ug.append(pair[1])
    ug = pd.DataFrame(ug)
    for l in ug.values:
        row = ''
        for a in l[1:]:
            row += str(a) + ' '
        lines.append(row)'''

    split = int(len(lines) * 0.9)
    lines_train = lines[:split]
    lines_valid = lines[split:]

    with open(unlabelled_train, "w") as train_output:
        for l in lines_train:
            train_output.write(l)
            train_output.write("\n")

    with open(unlabelled_valid, "w") as train_output:
        for l in lines_valid:
            train_output.write(l)
            train_output.write("\n")


base_dir = 'datasets' + os.sep


def get_datasets():
    datasets = [
        [('%sdirty_walmart_amazon/train.csv' % base_dir), ('%sdirty_walmart_amazon/tableA.csv' % base_dir),
         ('%sdirty_walmart_amazon/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
         'dirty_walmart_amazon',
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
        [('%swalmart_amazon/train.csv' % base_dir), ('%swalmart_amazon/tableA.csv' % base_dir),
         ('%swalmart_amazon/tableB.csv' % base_dir), [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], 'walmart_amazon',
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
    return datasets


def cheaper_train(dataset, params: CheapERParams):
    gt_file = dataset[0]
    t1_file = dataset[1]
    t2_file = dataset[2]
    indexes = dataset[3]
    dataset_name = dataset[4]
    flag_Anhai = dataset[6]
    seq_length = dataset[7]
    logging.info('CheapER: training on dataset "{}"'.format(dataset_name))
    logging.info('CheapER: using params "{}"'.format(params))
    return train_model(gt_file, t1_file, t2_file, indexes, dataset_name,
                       flag_Anhai, seq_length, params)
