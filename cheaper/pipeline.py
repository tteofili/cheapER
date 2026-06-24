from __future__ import print_function

import logging
import os
import random
import warnings
from datetime import date
from inspect import getsource
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from cheaper.data.create_datasets import add_shuffle, parse_original
from cheaper.data.create_datasets import create_datasets, add_identity, add_symmetry
from cheaper.data.csv2dataset import splitting_dataSet, parsing_anhai_nofilter, check_anhai_dataset, parse_unlabeled_pairs
from cheaper.data.plot import plot_graph
from cheaper.emt.emtmodel import EMTERModel
from cheaper.emt.logging_customized import setup_logging
from cheaper.params import CheapERParams
from cheaper.similarity import sim_function
from cheaper.similarity.similarity_utils import learn_best_aggregate
from cheaper.active_learning.first_iteration import first_iteration_select
from cheaper.oracle import get_oracle, adaptive_threshold, ADAPTIVE_THRESHOLD_METHODS

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
]

get_lambda_name = lambda l: getsource(l).strip()

setup_logging()


def sim_eval(simf, theta_min, theta_max, test):
    labels = []
    predicted_class = []
    for line in test:
        label = line[3]
        labels.append(label)
        # prediction = simf(line[0], line[1])[0]
        prediction = line[2]
        if prediction > theta_max:
            prediction = 1
        elif prediction < theta_min:
            prediction = 0
        else:
            if abs(prediction - theta_max) > abs(prediction - theta_min):
                prediction = 0
            else:
                prediction = 1
        predicted_class.append(prediction)
    result = classification_report(labels, predicted_class)

    l0 = result.split('\n')[2].split('       ')[2].split('      ')
    l1 = result.split('\n')[3].split('       ')[2].split('      ')
    p = l1[0]
    r = l1[1]
    f1 = l1[2]
    pnm = l0[0]
    rnm = l0[1]
    f1nm = l0[2]
    return p, r, f1, pnm, rnm, f1nm


def train_model(gt_file, t1_file, t2_file, indexes, dataset_name, flag_anhai, seq_length, params: CheapERParams):
    results = []

    tableA = pd.read_csv(t1_file)
    tableB = pd.read_csv(t2_file)

    basedir = 'models/' + dataset_name
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    unlabelled_train = basedir + '/unlabelled_train.txt'
    unlabelled_valid = basedir + '/unlabelled_valid.txt'

    for n in range(params.num_runs):
        for cut in params.slicing:
            if params.model_type == 'noisy-student':
                logging.info('Generating dataset')
                test_file = base_dir + dataset_name + os.sep + 'test.csv'
                valid_file = base_dir + dataset_name + os.sep + 'valid.csv'

                if getattr(params, 'training_data_source', 'ground_truth') == 'battleship_oracle':
                    unlabeled_path = getattr(params, 'unlabeled_train_csv', None) or (
                        base_dir + dataset_name + os.sep + 'train.csv'
                    )
                    if not os.path.exists(unlabeled_path):
                        raise FileNotFoundError(
                            "Battleship mode requires unlabeled candidate CSV at %s or set params.unlabeled_train_csv"
                            % unlabeled_path
                        )
                    candidates = parse_unlabeled_pairs(
                        unlabeled_path, t1_file, t2_file, indexes, max_candidates=-1
                    )
                    if not candidates:
                        raise ValueError("No candidate pairs loaded from %s" % unlabeled_path)
                    budget = getattr(params, 'initial_budget', 100)
                    strategy = getattr(params, 'first_iter_strategy', 'random')
                    seed = getattr(params, 'first_iter_seed', 42)
                    selected_indices = first_iteration_select(
                        candidates,
                        budget,
                        strategy=strategy,
                        seed=seed,
                        criterion=getattr(params, 'centrality_criterion', 'pagerank'),
                        nn_param=getattr(params, 'centrality_nn_param', 15),
                    )
                    oracle_threshold = getattr(params, 'oracle_threshold', 0.5)
                    oracle_fn = get_oracle(
                        getattr(params, 'oracle_type', 'similarity'),
                        threshold=oracle_threshold,
                        sim_function=simfunctions[0],
                    )
                    train_cut = []
                    matches = 0
                    non_matches = 0
                    use_adaptive = (
                        isinstance(oracle_threshold, str)
                        and oracle_threshold in ADAPTIVE_THRESHOLD_METHODS
                        and hasattr(oracle_fn, 'score')
                    )
                    if use_adaptive:
                        scores_list = []
                        for idx in selected_indices:
                            t1, t2 = candidates[idx]
                            scores_list.append(oracle_fn.score(t1, t2))
                        th = adaptive_threshold(scores_list, method=oracle_threshold)
                        logging.info('Adaptive oracle threshold (%s): %.4f', oracle_threshold, th)
                        for idx, score in zip(selected_indices, scores_list):
                            t1, t2 = candidates[idx]
                            label = 1 if score >= th else 0
                            if label == 1:
                                matches += 1
                            else:
                                non_matches += 1
                            train_cut.append((t1, t2, label))
                    else:
                        for idx in selected_indices:
                            t1, t2 = candidates[idx]
                            label = oracle_fn(t1, t2)
                            if int(label) == 1:
                                matches += 1
                            else:
                                non_matches += 1
                            train_cut.append((t1, t2, label))
                    # Balance so teacher gets equal matches and non-matches
                    positives = [p for p in train_cut if p[2] == 1]
                    negatives = [p for p in train_cut if p[2] == 0]
                    k = min(len(positives), len(negatives))
                    if k > 0:
                        random.seed(seed)
                        balanced = random.sample(positives, k) + random.sample(negatives, k)
                        random.shuffle(balanced)
                        train_cut = balanced
                        logging.info(
                            'Battleship initial labeling: %d candidates, selected %d, labeled by oracle -> %d raw (matches: %d, non-matches: %d) -> %d balanced train pairs (%d each)',
                            len(candidates), len(selected_indices), matches + non_matches, matches, non_matches,
                            len(train_cut), k,
                        )
                    else:
                        logging.warning(
                            'Battleship initial labeling: cannot balance (matches=%d, non_matches=%d); using all %d train pairs',
                            len(positives), len(negatives), len(train_cut),
                        )
                    _, test, valid = parse_original(
                        gt_file, t1_file, t2_file, indexes, simfunctions[0], flag_anhai,
                        valid_file, test_file, params.deeper_trick, cut=1
                    )
                else:
                    train, test, valid = parse_original(
                        gt_file, t1_file, t2_file, indexes, simfunctions[0], flag_anhai,
                        valid_file, test_file, params.deeper_trick, cut=cut
                    )
                    train_cut = train.copy()

                if params.seq_length > 0:
                    # override default dataset seq_length
                    seq_length = params.seq_length

                for model_type in params.models:

                    if params.compare:
                        train_base, test_base, valid_base = parse_original(gt_file, t1_file, t2_file, indexes, simfunctions[0],
                                                                           flag_anhai, valid_file, test_file, False, cut=cut)

                        train_cut_base = train_base.copy()
                        basic = EMTERModel(model_type)
                        basic.train(train_cut_base, valid_base, model_type, dataset_name,
                                  warmup=params.warmup, hf_training=params.hf_training, seq_length=seq_length,
                                  epochs=params.epochs, lr=params.lr * params.lr_multiplier, batch_size=params.batch_size,
                                  silent=params.silent, adaptive_ft=False, best_model=params.best_model,
                                  weight_decay=params.weight_decay, label_smoothing=params.label_smoothing)
                        pm, rm, f1m, pnm, rnm, f1nm = basic.eval(test_base, dataset_name, seq_length=seq_length,
                                                                 batch_size=params.batch_size, silent=params.silent)
                        new_row = {'model_type': model_type, 'train': 'baseline', 'cut': cut, 'pM': pm, 'rM': rm,
                                   'f1M': f1m, 'pNM': pnm, 'rNM': rnm, 'f1NM': f1nm}
                        results.append(new_row)

                    teacher = EMTERModel(model_type)

                    if params.adaptive_ft:
                        generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB)
                        teacher.adaptive_ft(unlabelled_train, unlabelled_valid, dataset_name, model_type,
                                            seq_length=seq_length, epochs=params.epochs, lr=params.lr)
                    logging.info("------------- Teacher Training {} ------------------".format(model_type))
                    if getattr(params, 'training_data_source', 'ground_truth') == 'battleship_oracle':
                        logging.info('Training with %d record pairs (Battleship oracle-labeled)', len(train_cut))
                    else:
                        logging.info('Training with {} record pairs ({}% GT)'.format(len(train_cut), 100 * cut))
                    teacher.train(train_cut, valid, model_type, dataset_name, seq_length=seq_length,
                                  warmup=params.warmup, hf_training=params.hf_training,
                                  epochs=params.epochs, lr=params.lr, batch_size=params.batch_size,
                                  silent=params.silent, adaptive_ft=params.adaptive_ft, best_model=params.best_model,
                                  weight_decay=params.weight_decay, label_smoothing=params.label_smoothing)
                    classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = teacher \
                        .eval(test, dataset_name, seq_length=seq_length, batch_size=params.batch_size,
                              silent=params.silent)
                    new_row = {'model_type': model_type, 'train': 'teacher', 'cut': cut, 'pM': classic_precision,
                               'rM': classic_recall,
                               'f1M': classic_f1,
                               'pNM': classic_precisionNOMATCH, 'rNM': classic_recallNOMATCH,
                               'f1NM': classic_f1NOMATCH}
                    results.append(new_row)
                    vinsim_data_app = []
                    threshold = 0.5
                    best_f1 = 0
                    sigma = params.sigma
                    kappa = params.kappa
                    for t_i in range(params.teaching_iterations):
                        if params.discard_old_data:
                            vinsim_data_app = []
                        temperature = 1
                        if params.temperature is not None:  # harder temperature leads to softer distributions
                            # inspired by rankmax, we adapt the temperature using the label approximation threshold (asc/desc)
                            if params.temperature == 'asc':
                                temperature = 1 +  t_i * (1 + threshold)
                            elif params.temperature == 'desc':
                                temperature = 1 -  t_i * (1 + threshold)
                            elif params.temperature == 'linear':
                                temperature = 1 + t_i
                            elif isinstance(params.temperature, float):
                                # see https://storage.googleapis.com/pub-tools-public-publication-data/pdf/87fc0a222b8e175c960e9ff391531cd977dfca35.pdf
                                temperature = params.temperature
                            else:
                                logging.warning(f'temperature param "{params.temperature}" set to 1')
                                temperature = 1
                        if params.mcd_samples > 1:
                            simf = lambda t1, t2: [np.stack([teacher.predict(t1, t2, t=temperature)['scores'].values[0] for _ in range(params.mcd_samples)]).mean(axis=0)]
                            teacher.enable_mcd()
                        else:
                            simf = lambda t1, t2: [teacher.predict(t1, t2, t=temperature)['scores'].values[0]] # score assegnato dal modello alla coppia

                        logging.info('Generating dataset')
                        # create datasets
                        test_file = base_dir + dataset_name + os.sep + 'test.csv'
                        valid_file = base_dir + dataset_name + os.sep + 'valid.csv'
                        if params.adjust_ds_size:
                            sigma = len(train_cut) * (2 + t_i)
                            kappa = int(sigma / 10)
                        _, _, _, _, vinsim_data_c, vinsim_data_app_c, threshold = create_datasets(gt_file,
                                                                                                  t1_file,
                                                                                                  t2_file,
                                                                                                  indexes, simf,
                                                                                                  dataset_name,
                                                                                                  sigma,
                                                                                                  flag_anhai,
                                                                                                  params.epsilon,
                                                                                                  kappa,
                                                                                                  params.num_runs,
                                                                                                  cut,
                                                                                                  valid_file,
                                                                                                  test_file,
                                                                                                  params.balance,
                                                                                                  params.deeper_trick,
                                                                                                  params.consistency,
                                                                                                  params.sim_edges,
                                                                                                  params.simple_slicing,
                                                                                                  margin_score=params.threshold)

                        logging.info('Previous generated dataset size: {}'.format(len(vinsim_data_app)))

                        if params.use_scores:
                            generated_data = vinsim_data_c
                        else:
                            generated_data = vinsim_data_app_c

                        for line in generated_data:
                            if line not in vinsim_data_app:
                                if params.sample_tag:
                                    line = (['PL '] + line[0], line[1], line[2])
                                vinsim_data_app += [line]

                        logging.info('New generated dataset size: {}'.format(len(vinsim_data_app)))

                        student = EMTERModel(model_type, model_noise=params.model_noise,
                                             add_layers=t_i * params.layers_increase)

                        logging.info("------------- Student Training {} -----------------".format(model_type))

                        dataDa = vinsim_data_app

                        if params.identity:
                            dataDa = add_identity(dataDa)

                        if params.symmetry:
                            dataDa = add_symmetry(dataDa)

                        if params.attribute_shuffle:
                            dataDa = add_shuffle(dataDa)

                        new_train = train_cut + dataDa

                        # gt+generated data train
                        logging.info(
                            'Training with {} record pairs ({} generated, {} GT)'.format(len(new_train), len(dataDa),
                                                                                         len(train_cut)))

                        if params.data_noise:
                            # add noise
                            for i in range(int(len(new_train) / 5)):
                                random_index = random.randint(0, len(new_train) - 1)
                                line = new_train[random_index]
                                rec_idx = random.randint(0, 1)
                                if rec_idx == 0:
                                    noised_line = (teacher.noise(line[rec_idx], mask=params.mask_token), line[1], line[2])
                                else:
                                    noised_line = (line[0], teacher.noise(line[rec_idx], mask=params.mask_token), line[2])
                                if params.sample_tag:
                                    noised_line = (['NS '] + noised_line[0], noised_line[1], noised_line[2])
                                new_train[random_index] = noised_line

                        student.train(new_train, valid, model_type, dataset_name, seq_length=seq_length,
                                      warmup=params.warmup, epochs=params.epochs + t_i, lr=params.lr * params.lr_multiplier,
                                      adaptive_ft=params.adaptive_ft, silent=params.silent, hf_training=params.hf_training,
                                      batch_size=params.batch_size, weight_decay=params.weight_decay,
                                      label_smoothing=params.label_smoothing, best_model=params.best_model,)
                        da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = student.eval(
                            test, dataset_name, seq_length=seq_length, batch_size=params.batch_size,
                            silent=params.silent)

                        if da_f1 > best_f1:
                            best_f1 = da_f1
                            save_path = 'models' + os.sep + dataset_name + os.sep + 'best'
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            student.save(save_path)
                        new_row = {'model_type': model_type, 'train': 'student', 'cut': cut, 'pM': da_precision,
                                   'rM': da_recall,
                                   'f1M': da_f1,
                                   'pNM': da_precisionNOMATCH,
                                   'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                        results.append(new_row)
                        logging.info(results)
                        teacher = student

            elif params.model_type == 'bert':

                logging.info('Generating dataset')
                # create datasets
                test_file = base_dir + dataset_name + os.sep + 'test.csv'
                valid_file = base_dir + dataset_name + os.sep + 'valid.csv'

                train, test, valid = parse_original(gt_file, t1_file, t2_file, indexes, simfunctions[0], flag_anhai,
                                                    valid_file, test_file, params.deeper_trick)

                train_cut = splitting_dataSet(cut, train)

                for model_type in params.models:

                    logging.info("------------- Vanilla EMT Training {} ------------------".format(model_type))
                    logging.info('Training with {} record pairs ({}% GT)'.format(len(train_cut), 100 * cut))
                    model = EMTERModel(model_type)

                    if params.adaptive_ft:
                        generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB)
                        model.adaptive_ft(unlabelled_train, unlabelled_valid, dataset_name, model_type,
                                          seq_length=seq_length,
                                          epochs=params.epochs, lr=min(5, params.epochs))

                    model.train(train_cut, valid, test, model_type, seq_length=seq_length, warmup=params.warmup,
                                epochs=params.epochs, lr=params.lr, batch_size=params.batch_size, silent=params.silent)
                    classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = model \
                        .eval(test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                    new_row = {'model_type': model_type, 'train': 'cl', 'cut': cut, 'pM': classic_precision,
                               'rM': classic_recall,
                               'f1M': classic_f1,
                               'pNM': classic_precisionNOMATCH, 'rNM': classic_recallNOMATCH,
                               'f1NM': classic_f1NOMATCH}
                    results.append(new_row)

                    simf = lambda t1, t2: [model.predict(t1, t2)['scores'].values[0]]

                    logging.info('Generating dataset')
                    # create datasets
                    test_file = base_dir + dataset_name + os.sep + 'test.csv'
                    valid_file = base_dir + dataset_name + os.sep + 'valid.csv'
                    data, train, valid, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file,
                                                                                             t2_file, indexes, simf,
                                                                                             dataset_name,
                                                                                             params.sigma,
                                                                                             flag_anhai, params.epsilon,
                                                                                             params.kappa,
                                                                                             params.num_runs, cut,
                                                                                             valid_file,
                                                                                             test_file, params.balance,
                                                                                             params.adjust_ds_size,
                                                                                             params.deeper_trick,
                                                                                             params.consistency,
                                                                                             params.sim_edges,
                                                                                             params.simple_slicing,
                                                                                             margin_score=.5)
                    logging.info('Generated dataset size: {}'.format(len(vinsim_data_app)))

                    model = EMTERModel(model_type)

                    logging.info("------------- Data augmented EMT Training {} -----------------".format(model_type))

                    dataDa = vinsim_data_app

                    if params.identity:
                        dataDa = add_identity(dataDa)

                    if params.symmetry:
                        dataDa = add_symmetry(dataDa)

                    if params.attribute_shuffle:
                        dataDa = add_shuffle(dataDa)

                    # generated data train only
                    if params.generated_only:
                        model.train(dataDa, valid, model_type, dataset_name, seq_length=seq_length,
                                    warmup=params.warmup,
                                    epochs=params.epochs, lr=params.lr, adaptive_ft=params.adaptive_ft,
                                    silent=params.silent,
                                    batch_size=params.batch_size)

                        da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(
                            test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                        new_row = {'model_type': model_type, 'train': 'da-only', 'cut': cut, 'pM': da_precision,
                                   'rM': da_recall,
                                   'f1M': da_f1,
                                   'pNM': da_precisionNOMATCH,
                                   'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                        results.append(new_row)

                    # gt+generated data train
                    model = EMTERModel(model_type)
                    model.train(train_cut + dataDa, valid, model_type, dataset_name, seq_length=seq_length,
                                warmup=params.warmup,
                                epochs=params.epochs, lr=params.lr, adaptive_ft=params.adaptive_ft,
                                silent=params.silent,
                                batch_size=params.batch_size)

                    da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(
                        test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                    new_row = {'model_type': model_type, 'train': 'da', 'cut': cut, 'pM': da_precision, 'rM': da_recall,
                               'f1M': da_f1,
                               'pNM': da_precisionNOMATCH,
                               'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                    results.append(new_row)

                    logging.info(results)
            elif params.model_type == 'hybrid':
                simf = learn_best_aggregate(gt_file, t1_file, t2_file, indexes, simfunctions, cut, params.sim_length,
                                            normalize=params.normalize, lm=params.approx,
                                            deeper_trick=params.deeper_trick)

                logging.info('Generating dataset')
                # create datasets
                test_file = base_dir + dataset_name + os.sep + 'test.csv'
                valid_file = base_dir + dataset_name + os.sep + 'valid.csv'
                data, train, valid, test, vinsim_data, vinsim_data_app = create_datasets(gt_file, t1_file,
                                                                                         t2_file, indexes, simf,
                                                                                         dataset_name,
                                                                                         params.sigma,
                                                                                         flag_anhai, params.epsilon,
                                                                                         params.kappa,
                                                                                         params.num_runs, cut,
                                                                                         valid_file,
                                                                                         test_file, params.balance,
                                                                                         params.adjust_ds_size,
                                                                                         params.deeper_trick,
                                                                                         params.consistency,
                                                                                         params.sim_edges,
                                                                                         params.simple_slicing)
                logging.info('Generated dataset size: {}'.format(len(vinsim_data_app)))

                generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB)

                train_cut = splitting_dataSet(cut, train)

                for model_type in params.models:

                    if params.compare:
                        logging.info("------------- Vanilla EMT Training {} ------------------".format(model_type))
                        logging.info('Training with {} record pairs ({}% GT)'.format(len(train_cut), 100 * cut))
                        model = EMTERModel(model_type)

                        model.train(train_cut, valid, test, model_type, seq_length=seq_length, warmup=params.warmup,
                                    epochs=params.epochs, lr=params.lr, batch_size=params.batch_size,
                                    silent=params.silent)
                        classic_precision, classic_recall, classic_f1, classic_precisionNOMATCH, classic_recallNOMATCH, classic_f1NOMATCH = model \
                            .eval(test, dataset_name, seq_length=seq_length, batch_size=params.batch_size,
                                  silent=params.silent)
                        new_row = {'model_type': model_type, 'train': 'cl', 'cut': cut, 'pM': classic_precision,
                                   'rM': classic_recall,
                                   'f1M': classic_f1,
                                   'pNM': classic_precisionNOMATCH, 'rNM': classic_recallNOMATCH,
                                   'f1NM': classic_f1NOMATCH}
                        results.append(new_row)

                    logging.info("------------- Data augmented EMT Training {} -----------------".format(model_type))

                    dataDa = vinsim_data_app

                    if params.identity:
                        dataDa = add_identity(dataDa)

                    if params.symmetry:
                        dataDa = add_symmetry(dataDa)

                    if params.attribute_shuffle:
                        dataDa = add_shuffle(dataDa)

                    model = EMTERModel(model_type)

                    if params.adaptive_ft:
                        model.adaptive_ft(unlabelled_train, unlabelled_valid, dataset_name, model_type,
                                          seq_length=seq_length,
                                          epochs=params.epochs, lr=params.lr)

                    # generated data train only
                    if params.generated_only:
                        model.train(dataDa, valid, model_type, dataset_name, seq_length=seq_length,
                                    warmup=params.warmup,
                                    epochs=params.epochs, lr=params.lr, adaptive_ft=params.adaptive_ft,
                                    silent=params.silent,
                                    batch_size=params.batch_size)

                        da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(
                            test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                        new_row = {'model_type': model_type, 'train': 'da-only', 'cut': cut, 'pM': da_precision,
                                   'rM': da_recall,
                                   'f1M': da_f1,
                                   'pNM': da_precisionNOMATCH,
                                   'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                        results.append(new_row)

                    # gt+generated data train
                    model = EMTERModel(model_type)
                    model.train(train_cut + dataDa, valid, model_type, dataset_name, seq_length=seq_length,
                                warmup=params.warmup,
                                epochs=params.epochs, lr=params.lr, adaptive_ft=params.adaptive_ft,
                                silent=params.silent,
                                batch_size=params.batch_size)

                    da_precision, da_recall, da_f1, da_precisionNOMATCH, da_recallNOMATCH, da_f1NOMATCH = model.eval(
                        test, dataset_name, seq_length=seq_length, batch_size=params.batch_size)
                    new_row = {'model_type': model_type, 'train': 'da', 'cut': cut, 'pM': da_precision, 'rM': da_recall,
                               'f1M': da_f1,
                               'pNM': da_precisionNOMATCH,
                               'rNM': da_recallNOMATCH, 'f1NM': da_f1NOMATCH}
                    results.append(new_row)

                    logging.info(results)

            elif params.model_type == 'sims':
                simf = learn_best_aggregate(gt_file, t1_file, t2_file, indexes, simfunctions, cut, params.sim_length,
                                            normalize=params.normalize, lm=params.approx,
                                            deeper_trick=params.deeper_trick)
                if params.deeper_trick:
                    train_data = check_anhai_dataset(gt_file, t1_file, t2_file, indexes, simf)
                else:
                    train_data = parsing_anhai_nofilter(gt_file, t1_file, t2_file, indexes, simf)
                tplus, tmin = plot_graph(train_data, cut)

                theta_min = min(tplus, tmin)
                theta_max = max(tplus, tmin)

                theta_max = min(theta_max + params.epsilon, 0.95)

                test_file = base_dir + dataset_name + os.sep + 'test.csv'
                test_data = parsing_anhai_nofilter(test_file, t1_file, t2_file, indexes, simf)
                sim_precision, sim_recall, sim_f1, sim_precisionNOMATCH, sim_recallNOMATCH, sim_f1NOMATCH = sim_eval(
                    simf, theta_min, theta_max, test_data)

                new_row = {'model_type': 'sims', 'train': params.approx, 'cut': cut, 'pM': sim_precision,
                           'rM': sim_recall,
                           'f1M': sim_f1,
                           'pNM': sim_precisionNOMATCH,
                           'rNM': sim_recallNOMATCH, 'f1NM': sim_f1NOMATCH}
                results.append(new_row)

                logging.info(results)

        today = date.today()
        filename = 'results' + os.sep + today.strftime("%b-%d-%Y") + '_' + dataset_name + '.csv'
        with open(filename, 'a') as f:
            f.write('# ' + str(params) + '\n')
        pd.DataFrame(results).to_csv(filename, mode='a')
    return results


def get_row(r1, r2, lprefix='ltable_', rprefix='rtable_'):
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    return r1r2


def generate_unlabelled(unlabelled_train, unlabelled_valid, tableA, tableB):
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
    # Each row: [gt_file, t1_file, t2_file, indexes, dataset_name, temp_dir, flag_anhai, seq_length]
    # Optional 9th element: unlabeled_train_csv for Battleship mode (candidate pairs CSV with id1, id2)
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
         ('%stemporary/' % base_dir), True, 275],
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
    flag_anhai = dataset[6]
    seq_length = dataset[7]
    if len(dataset) > 8 and dataset[8] is not None:
        params.unlabeled_train_csv = dataset[8]
    logging.info('CheapER: training on dataset "{}"'.format(dataset_name))
    logging.info('CheapER: using params "{}"'.format(params))
    if params.silent:
        warnings.filterwarnings("ignore")
    return train_model(gt_file, t1_file, t2_file, indexes, dataset_name, flag_anhai, seq_length, params)
