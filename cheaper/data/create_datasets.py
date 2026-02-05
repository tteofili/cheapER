from __future__ import print_function

import logging
import random
from random import shuffle

from cheaper.data.csv2dataset import csv_2_dataset_alternate, parsing_anhai_nofilter, check_anhai_dataset
from cheaper.data.plot import plot_occurrence, plot_pretrain, plot_data_pt, plot_graph
from cheaper.data.sampling_dataset_pt import create_lists
from cheaper.data.test_occ_attr import init_dict_lista
from cheaper.emt.logging_customized import setup_logging
from cheaper.similarity.sim_function import min_cos

setup_logging()


def add_symmetry(dataDa):
    new_list = dataDa[:]
    for pair in dataDa:
        t1 = pair[0][:]
        t2 = pair[1][:]
        label = pair[2]
        new_list.append((t2, t1, label))
    return new_list


def add_identity(dataDa):
    new_list = dataDa[:]
    for pair in dataDa:
        t1 = pair[0][:]
        t2 = pair[1][:]
        new_list.append((t1, t1, 1))
        new_list.append((t2, t2, 1))
    return new_list


def add_shuffle(dataDa, mult: int = 1):
    new_list = dataDa[:]
    for pair in dataDa:
        t1 = pair[0][:]
        t2 = pair[1][:]
        label = pair[2]
        for a_idx in range(mult):
            random.shuffle(t1)
            random.shuffle(t2)
            new_list.append((t1, t2, label))
    return new_list

def parse_original(ground_truth_file, table1_file, table2_file, att_indexes, simf, flag_anhai, valid_file, test_file,
                   deeper_trick, cut=1):
    logging.info('Parsing original dataset')
    if flag_anhai == False:
        data = csv_2_dataset_alternate(ground_truth_file, table1_file, table2_file, att_indexes, simf, cut=cut)
        valid_data = csv_2_dataset_alternate(valid_file, table1_file, table2_file, att_indexes, simf)
        test_data = csv_2_dataset_alternate(test_file, table1_file, table2_file, att_indexes, simf)
    else:
        if deeper_trick:
            data = check_anhai_dataset(ground_truth_file, table1_file, table2_file, att_indexes, simf, cut=cut)
        else:
            data = parsing_anhai_nofilter(ground_truth_file, table1_file, table2_file, att_indexes, simf, cut=cut)
        valid_data = parsing_anhai_nofilter(valid_file, table1_file, table2_file, att_indexes, simf)
        test_data = parsing_anhai_nofilter(test_file, table1_file, table2_file, att_indexes, simf)

    data = list(map(lambda q: (q[0], q[1], q[3]), data))
    valid_data = list(map(lambda q: (q[0], q[1], q[3]), valid_data))
    test_data = list(map(lambda q: (q[0], q[1], q[3]), test_data))
    return data, test_data, valid_data


def create_datasets(ground_truth_file, table1_file, table2_file, att_indexes, simf, dataset_name, tot_pt, flag_anhai,
                    epsilon, tot_copy, num_run, cut, valid_file, test_file, balance, deeper_trick,
                    consistency, sim_edges, simple_slicing, margin_score=0):
    logging.info('Parsing original dataset')
    if flag_anhai == False:
        data = csv_2_dataset_alternate(ground_truth_file, table1_file, table2_file, att_indexes, simf)
        valid_data = csv_2_dataset_alternate(valid_file, table1_file, table2_file, att_indexes, simf)
        test_data = csv_2_dataset_alternate(test_file, table1_file, table2_file, att_indexes, simf)
    else:
        if deeper_trick:
            data = check_anhai_dataset(ground_truth_file, table1_file, table2_file, att_indexes, simf)
        else:
            data = parsing_anhai_nofilter(ground_truth_file, table1_file, table2_file, att_indexes, simf)
        valid_data = parsing_anhai_nofilter(valid_file, table1_file, table2_file, att_indexes, simf)
        test_data = parsing_anhai_nofilter(test_file, table1_file, table2_file, att_indexes, simf)

    min_sim_match, max_sim_no_match = plot_graph(data, cut)
    logging.info("min_sim_match " + str(min_sim_match) + "max_sim_no_match " + str(max_sim_no_match))
    if margin_score > 0:
        max_sim = margin_score + epsilon
        min_sim = margin_score - epsilon
    else:
        max_sim = min(epsilon + max(min_sim_match, max_sim_no_match), 0.99999)
        min_sim = max(min(min_sim_match, max_sim_no_match) - epsilon, 0.000001)
    if min_sim > 0.5:
        min_sim = 0.5
    if max_sim < 0.5:
        max_sim = 0.5

    logging.info("!max_sim " + str(max_sim))
    logging.info("!min_sim " + str(min_sim))
    # Dataset per DeepER classico: [(tupla1, tupla2, label), ...].
    deeper_data = list(map(lambda q: (q[0], q[1], q[3]), data))
    deeper_valid_data = list(map(lambda q: (q[0], q[1], q[3]), valid_data))
    deeper_test_data = list(map(lambda q: (q[0], q[1], q[3]), test_data))

    # Tutti i successivi addestramenti partiranno dal 100% di deeper_train (80% di tutti i dati).
    # Le tuple in deeper_test non verranno mai usate per addestrare ma solo per testare i modelli.
    deeper_train = deeper_data
    deeper_valid = deeper_valid_data
    deeper_test = deeper_test_data

    logging.info("--------------- Generating datasets --------------")
    # Costruzione Dataset
    k_slice = int(tot_pt // 2)  # quanti match e non match andranno a formare il dataset di PT

    vinsim_data = []

    # Preleva solo quelle in match con il relativo sim vector.
    for i in range(len(data)):
        if data[i][3] == 1:
            r = data[i]
            vinsim_data.append((r[0], r[1], r[2]))

    # Taglio della porzione desiderata.
    bound = int(len(data) * cut)
    vinsim_data = vinsim_data[:bound]

    min_cos_sim = min_cos(vinsim_data)
    logging.info("min_cos_sim " + str(min_cos_sim))

    max_occ = 8

    vinsim_data = []
    result_list_noMatch = []
    result_list_match = []
    consistency_list = []
    min_sim_c = min_sim
    max_sim_c = max_sim
    it = 0
    while len(result_list_match) < k_slice and len(result_list_match) < k_slice and it < 2:
        logging.info(f"creating data with theta_min:{min_sim_c}, theta_max:{max_sim_c}")
        result_list_noMatch, result_list_match, consistency_list = create_lists(table1_file, table2_file, tot_pt,
                                                                                min_sim_c,
                                                                                max_sim_c, att_indexes,
                                                                                min_cos_sim, tot_copy, max_occ,
                                                                                sim_function=simf)
        logging.info("{} matches, {} non-matches, {} consistency pairs".format(len(result_list_match),
                                                                               len(result_list_noMatch),
                                                                               len(consistency_list)))
        delta = (max_sim_c - min_sim)/10
        if (min_sim_c + delta) >= 0.5 or (max_sim_c - delta) <= 0.5:
            break
        else:
            max_sim_c = max_sim_c - delta
            min_sim_c = min_sim_c + delta
            max_sim_c = max(min_sim_c, max_sim_c)
            min_sim_c = min(min_sim_c, max_sim_c)
            it += 1


    min_sim = min_sim_c
    max_sim = max_sim_c

    # unione in una sola lista random_tuples0= insieme dei candidati per il pt
    random_tuples0 = result_list_noMatch + result_list_match

    logging.info("tot_pt: " + str(tot_pt))
    logging.info("len(random_tuples0) " + str(len(random_tuples0)))
    logging.info("len(result_list_noMatch) " + str(len(result_list_noMatch)))
    logging.info("len(result_list_match) " + str(len(result_list_match)))

    random.shuffle(random_tuples0)
    random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
    plot_pretrain(random_tuples0sort)

    k_slice_max = min(len(result_list_match), len(result_list_noMatch))

    if simple_slicing and k_slice_max > 0:
        k_slice = tot_pt // 2
        logging.info("k_slice {}".format(str(k_slice)))
        if k_slice > k_slice_max:
            k_slice = k_slice_max
        if k_slice == 0:
            k_slice = -1

        result_list_match = sorted(result_list_match, key=lambda tup: (tup[2][0]), reverse=sim_edges)
        result_list_noMatch = sorted(result_list_noMatch, key=lambda tup: (tup[2][0]), reverse=not sim_edges)

        neg_slice = int(k_slice * (0.5 + balance[0]))
        pos_slice = int(k_slice * (0.5 + balance[1]))
        if consistency:
            consistency_slice = len(result_list_match) + len(consistency_list) - len(result_list_noMatch)
            vinsim_data += consistency_list[:consistency_slice]
            logging.info("adding {} consistency pairs".format(len(consistency_list[:consistency_slice])))

        non_matching_candidates = result_list_noMatch[:neg_slice]
        logging.info("adding {} non-matching pairs".format(len(non_matching_candidates)))
        random_tuples1 = non_matching_candidates  # likely non matches

        matching_candidates = result_list_match[-pos_slice:]
        logging.info("adding {} matching pairs".format(len(matching_candidates)))
        random_tuples2 = matching_candidates  # likely matches

        vinsim_data += random_tuples1
        vinsim_data += random_tuples2
        logging.info("generated data size {}".format(len(vinsim_data)))
    else:
        k_slice = min(len(result_list_match), len(result_list_noMatch))
        if k_slice == 0:
            k_slice = -1

        neg_slice = int(k_slice * (0.5 + balance[0]))
        if sim_edges:
            random_tuples1 = sorted(result_list_noMatch, key=lambda tup: (tup[2][0]))[:neg_slice]  # likely non matches
        else:
            random_tuples1 = sorted(result_list_noMatch, key=lambda tup: (tup[2][0]))[-neg_slice:]  # likely non matches
        logging.info("num of non-matches {}".format(len(random_tuples1)))

        pos_slice = int(k_slice * (0.5 + balance[1]))
        if sim_edges:
            random_tuples2 = sorted(result_list_match, key=lambda tup: (tup[2][0]))[-pos_slice:]  # likely matches
        else:
            random_tuples2 = sorted(result_list_match, key=lambda tup: (tup[2][0]))[:pos_slice]  # likely matches
        logging.info("num of matches {}".format(len(random_tuples2)))
        if not consistency and len(random_tuples1) < tot_pt:
            consistency_slice = tot_pt - len(random_tuples1)
            logging.info("adding {} consistency pairs".format(consistency_slice))
            random_tuples1 += consistency_list[:consistency_slice]
        elif consistency:
            logging.info("adding {} consistency pairs".format(len(consistency_list)))
            random_tuples1 += consistency_list

        random_tuples1 += random_tuples2

        logging.debug(len(random_tuples1))
        # Concatenazione.
        vinsim_data += random_tuples1

    # Shuffle.
    shuffle(vinsim_data)

    # plotting del dataset di pt finale
    plot_data_pt(vinsim_data)

    logging.info("--------------- data augmentation creating dataset --------------")

    # arrotonda il sim_value a 0/1 per il test di data_augmentation
    def convert_approx(tuples, min_t=0.5, max_t=0.5):
        round_list = []
        discarded = []
        for el in tuples:
            if el[2][0] >= max_t:
                sim_value = 1
                round_list.append((el[0], el[1], [sim_value]))
            elif el[2][0] <= min_t:
                sim_value = 0
                round_list.append((el[0], el[1], [sim_value]))
            else:
                discarded.append((el[0], el[1], [el[2][0]]))
        return round_list, discarded

    # vinsim_data_app è il dataset di pt approssimato a 0/1
    logging.info(f'using threshold={max_sim} to approximate label')
    vinsim_data_app, ignored = convert_approx(vinsim_data, min_t=min_sim, max_t=max_sim)
    logging.info('discarded {} elements'.format(len(ignored)))
    logging.debug(vinsim_data_app[:15])

    plot_data_pt(vinsim_data_app)

    # Salva dataset su disco.
    with open('datasets/temporary/datasetPT_{a}_{b}.txt'.format(a=dataset_name, b=num_run), 'w') as output:
        output.write(str(vinsim_data_app))

    # Dataset per il test di data_augmentation: [(tupla1, tupla2, label), ...]
    # VANNO AGGIUNTI I TAGLI DELLA Ground Truth [200,100,50...] in ogni addestramento
    vinsim_data_app = list(map(lambda q: (q[0], q[1], q[2][0]), vinsim_data_app))

    epsilon = max_sim
    return data, deeper_train, deeper_valid, deeper_test, vinsim_data, vinsim_data_app, epsilon
