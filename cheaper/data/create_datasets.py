from __future__ import print_function
import random
from collections import Counter
from cheaper.data.csv2dataset import csv_2_datasetALTERNATE, parsing_anhai_dataOnlyMatch, parsing_anhai_nofilter, \
    check_anhai_dataset
from cheaper.data.plot import plotting_occorrenze, plot_pretrain, plot_dataPT, plot_graph
from cheaper.data.sampling_dataset_pt import csvTable2datasetRANDOM_countOcc
from cheaper.data.test_occ_attr import init_dict_lista
from cheaper.similarity.sim_function import min_cos
from random import shuffle
import logging
from cheaper.emt.logging_customized import setup_logging

setup_logging()

def create_datasets(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf, DATASET_NAME, tot_pt, flag_Anhai,
                    soglia, tot_copy, num_run, cut, valid_file, test_file):
    logging.info('Parsing original dataset')
    if flag_Anhai == False:
        data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        valid_data = csv_2_datasetALTERNATE(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = csv_2_datasetALTERNATE(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
    else:
        data = check_anhai_dataset(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        # data = parsing_anhai_nofilter(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        valid_data = parsing_anhai_nofilter(valid_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)
        test_data = parsing_anhai_nofilter(test_file, TABLE1_FILE, TABLE2_FILE, ATT_INDEXES, simf)

    min_sim_Match, max_sim_noMatch = plot_graph(data, cut)
    logging.info("min_sim_Match " + str(min_sim_Match) + "max_sim_noMatch " + str(max_sim_noMatch))
    max_sim = soglia + max(min_sim_Match, max_sim_noMatch)
    if max_sim > 0.9:
        max_sim = 0.9
    logging.info("!max_sim " + str(max_sim))
    min_sim = min(min_sim_Match, max_sim_noMatch)  # -soglia
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
    bound = int(len(vinsim_data) * cut)
    vinsim_data = vinsim_data[:bound]

    min_cos_sim = min_cos(vinsim_data)
    logging.info("min_cos_sim " + str(min_cos_sim))

    max_occ = 8

    vinsim_data = []
    # costruisce i dataset di pt con un max di occurrenza di una tuple di max_occ volte   csvTable2datasetRANDOM_NOOcc
    # tot_pt = max(1000, bound * 2)
    # tot_copy = tot_pt * 0.1
    result_list_noMatch, result_list_match = csvTable2datasetRANDOM_countOcc(TABLE1_FILE, TABLE2_FILE, 2 * tot_pt,
                                                                             min_sim,
                                                                             max_sim, ATT_INDEXES,
                                                                             min_cos_sim, tot_copy, max_occ, simf)

    result_list_noMatch = result_list_noMatch[:len(result_list_match)]
    # test per il count dei valori degli attributi
    lista_attrMATCH, lista_attrNO_MATCH = init_dict_lista(result_list_match, result_list_noMatch, len(ATT_INDEXES))
    logging.info("dizionari occorrenze degli attributi del dataset di pt")
    j = 0
    for dictionario in lista_attrMATCH:
        j = j + 1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrMATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            logging.info('%s: %i' % (k, v))
    j = 0
    for dictionario in lista_attrNO_MATCH:
        j = j + 1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrNO_MATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            logging.info('%s: %i' % (k, v))

    # fine test

    # unione in una sola lista random_tuples0= insieme dei candidati per il pt
    random_tuples0 = result_list_noMatch + result_list_match

    logging.info("tot_pt: " + str(tot_pt))
    # print("len(datapt_hash) " +str(len(datapt_hash)))
    logging.info("len(random_tuples0) " + str(len(random_tuples0)))
    logging.info("len(result_list_noMatch) " + str(len(result_list_noMatch)))
    logging.info("len(result_list_match) " + str(len(result_list_match)))

    random.shuffle(random_tuples0)
    random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
    logging.info("---------------- CANDIDATES RANDOM TUPLES -------------------------")
    plot_pretrain(random_tuples0sort)

    # SERVE per controllare che i match e i non match siano di egual numero
    # altrimenti si riduce il taglio di k
    # si suppone che sia riuscito a trovare meno match del k_slice=tot_pt/2
    if len(result_list_match) <= tot_pt / 2:
        k_slice = int(len(result_list_match))

        logging.info("riduco k")

    # print di alcuni elementidel dataset di pt e get k estremi che formeranno il dataset di pt
    logging.info("k_slice : " + str(k_slice))
    random_tuples1 = random_tuples0sort[:k_slice]
    logging.info("random_tuples1[:10]")
    logging.info(random_tuples1[:10])
    logging.info("random_tuples1[-10:]")
    logging.info(random_tuples1[-10:])
    random_tuples2 = random_tuples0sort[-k_slice:]
    logging.info("random_tuples2[:10]")
    logging.info(random_tuples2[:10])
    logging.info("random_tuples2[-10:]")
    logging.info(random_tuples2[-10:])

    random_tuples1 += random_tuples2

    logging.info(len(random_tuples1))
    # Concatenazione.
    vinsim_data += random_tuples1

    # vinsim_data +=random_tuples0
    # vinsim_data += random_tuples
    # Shuffle.
    shuffle(vinsim_data)

    # plotting del dataset di pt finale
    plot_dataPT(vinsim_data)

    logging.info("--------------- data augmentation creating dataset --------------")

    # arrotonda il sim_value a 0/1 per il test di data_augmentation
    def converti_approssima(tuples, min_t=0.5, max_t=0.5):
        round_list = []
        for el in tuples:
            if el[2][0] > max_t:
                sim_value = 1
                round_list.append((el[0], el[1], [sim_value]))
            elif el[2][0] < min_t:
                sim_value = 0
                round_list.append((el[0], el[1], [sim_value]))
        return round_list

    # vinsim_data_app è il dataset di pt approssimato a 0/1
    logging.info(f'using threshold={max_sim} to approximate label')
    vinsim_data_app = converti_approssima(vinsim_data, min_t=min_sim, max_t=max_sim)
    logging.info(vinsim_data_app[:15])

    # Filtro.
    vinsim_data_app = shrink_data(vinsim_data_app)

    plot_dataPT(vinsim_data_app)

    # Salva dataset su disco.
    with open('datasets/temporary/datasetPT_{a}_{b}.txt'.format(a=DATASET_NAME, b=num_run), 'w') as output:
        output.write(str(vinsim_data_app))

    # Dataset per il test di data_augmentation: [(tupla1, tupla2, label), ...]
    # VANNO AGGIUNTI I TAGLI DELLA Ground Truth [200,100,50...] in ogni addestramento
    vinsim_data_app = list(map(lambda q: (q[0], q[1], q[2][0]), vinsim_data_app))

    return data, deeper_train, deeper_valid, deeper_test, vinsim_data, vinsim_data_app
