from __future__ import print_function

import logging
import random
from collections import Counter
from random import shuffle

from cheaper.data.csv2dataset import csv_2_datasetALTERNATE, parsing_anhai_nofilter, check_anhai_dataset
from cheaper.data.plot import plotting_occorrenze, plot_pretrain, plot_dataPT, plot_graph
from cheaper.data.sampling_dataset_pt import csvTable2datasetRANDOM_countOcc, create_lists
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

def parse_original(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, flag_Anhai, valid_file, test_file,
                   deeper_trick, cut=1):
    logging.info('Parsing original dataset')
    if flag_Anhai == False:
        data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, cut=cut)
        valid_data = csv_2_datasetALTERNATE(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = csv_2_datasetALTERNATE(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
    else:
        if deeper_trick:
            data = check_anhai_dataset(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, cut=cut)
        else:
            data = parsing_anhai_nofilter(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, cut=cut)
        valid_data = parsing_anhai_nofilter(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = parsing_anhai_nofilter(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

    data = list(map(lambda q: (q[0], q[1], q[3]), data))
    valid_data = list(map(lambda q: (q[0], q[1], q[3]), valid_data))
    test_data = list(map(lambda q: (q[0], q[1], q[3]), test_data))
    return data, test_data, valid_data


def create_datasets(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, DATASET_NAME, tot_pt, flag_Anhai,
                    soglia, tot_copy, num_run, cut, valid_file, test_file, balance, deeper_trick,
                    consistency, sim_edges, simple_slicing, margin_score=0):
    logging.info('Parsing original dataset')
    if flag_Anhai == False:
        data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        valid_data = csv_2_datasetALTERNATE(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = csv_2_datasetALTERNATE(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
    else:
        if deeper_trick:
            data = check_anhai_dataset(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        else:
            data = parsing_anhai_nofilter(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        valid_data = parsing_anhai_nofilter(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = parsing_anhai_nofilter(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

    min_sim_Match, max_sim_noMatch = plot_graph(data, cut)
    logging.info("min_sim_Match " + str(min_sim_Match) + "max_sim_noMatch " + str(max_sim_noMatch))
    if margin_score > 0:
        max_sim = margin_score + soglia #max(max_sim_noMatch, margin_score)
        min_sim = margin_score - soglia #min(margin_score, min_sim_Match)
    else:
        max_sim = min(soglia + max(min_sim_Match, max_sim_noMatch), 0.99999)
        min_sim = max(min(min_sim_Match, max_sim_noMatch) - soglia, 0.000001)
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

    # Taglia attributi se troppo lunghi
    # Alcuni dataset hanno attributi con descrizioni molto lunghe.
    # Questo filtro limita il numero di caratteri di un attributo a 1000.
    def shrink_data(data):

        def cut_string(s):
            if len(s) >= 1000:
                return s[:1000]
            else:
                return s

        temp = []
        for t1, t2, lb in data:
            t1 = list(map(cut_string, t1))
            t2 = list(map(cut_string, t2))
            temp.append((t1, t2, lb))

        return temp

    deeper_data = shrink_data(deeper_data)
    deeper_valid_data = shrink_data(deeper_valid_data)
    deeper_test_data = shrink_data(deeper_test_data)

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
    # costruisce i dataset di pt con un max di occurrenza di una tuple di max_occ volte   csvTable2datasetRANDOM_NOOcc
    # tot_pt = max(1000, bound * 2)
    # tot_copy = tot_pt * 0.1
    min_sim_c = min_sim
    max_sim_c = max_sim
    it = 0
    while len(result_list_match) < tot_pt/2 and len(result_list_match) < tot_pt/2 and it < 2:
        logging.info(f"creating data with theta_min:{min_sim_c}, theta_max:{max_sim_c}")
        result_list_noMatch, result_list_match, consistency_list = create_lists(TABLE1_FILE, TABLE2_FILE, tot_pt,
                                                                                min_sim_c,
                                                                                max_sim_c, ATT_INDEXES,
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

    # result_list_noMatch = result_list_noMatch[:len(result_list_match)]
    # test per il count dei valori degli attributi
    lista_attrMATCH, lista_attrNO_MATCH = init_dict_lista(result_list_match, result_list_noMatch, len(ATT_INDEXES))
    logging.info("dizionari occorrenze degli attributi del dataset di pt")
    j = 0
    for dictionario in lista_attrMATCH:
        j = j + 1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrMATCH{j}')
        #d = Counter(dictionario)
        # for k, v in d.most_common(5):
        #     logging.info('%s: %i' % (k, v))
    j = 0
    for dictionario in lista_attrNO_MATCH:
        j = j + 1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrNO_MATCH{j}')
        #d = Counter(dictionario)
        # for k, v in d.most_common(5):
        #     logging.info('%s: %i' % (k, v))

    # fine test

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
    plot_dataPT(vinsim_data)

    logging.info("--------------- data augmentation creating dataset --------------")

    # arrotonda il sim_value a 0/1 per il test di data_augmentation
    def converti_approssima(tuples, min_t=0.5, max_t=0.5):
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
    vinsim_data_app, ignored = converti_approssima(vinsim_data, min_t=min_sim, max_t=max_sim)
    logging.info('discarded {} elements'.format(len(ignored)))
    logging.debug(vinsim_data_app[:15])

    # Filtro.
    vinsim_data_app = shrink_data(vinsim_data_app)

    plot_dataPT(vinsim_data_app)

    # Salva dataset su disco.
    with open('datasets/temporary/datasetPT_{a}_{b}.txt'.format(a=DATASET_NAME, b=num_run), 'w') as output:
        output.write(str(vinsim_data_app))

    # Dataset per il test di data_augmentation: [(tupla1, tupla2, label), ...]
    # VANNO AGGIUNTI I TAGLI DELLA Ground Truth [200,100,50...] in ogni addestramento
    vinsim_data_app = list(map(lambda q: (q[0], q[1], q[2][0]), vinsim_data_app))

    threshold = max_sim
    return data, deeper_train, deeper_valid, deeper_test, vinsim_data, vinsim_data_app, threshold
