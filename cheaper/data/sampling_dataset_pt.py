import csv
import logging
import math
import random
import re
import traceback
from collections import Counter

import numpy as np
from datasketch import MinHash, MinHashLSH
from scipy.stats import entropy

from cheaper.data.edit_dna import Sequence
from cheaper.data.plot import plot_dicts, plot_data_pt
from cheaper.emt.logging_customized import setup_logging

setup_logging()


WORD = re.compile(r'\w+')


# cosine similarity between two vectors
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def concatenate_list_data(list):
    result = ''
    for element in list:
        result += ' ' + str(element)
    return result


def entropy1(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def create_data(tableL, tableR, indiciL, indiciR):
    table1 = csv.reader(open(tableL, encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR, encoding="utf8"), delimiter=',')
    next(table1, None)
    next(table2, None)

    # convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)

    result_list = []
    result_list1, dataL = sampling_table(tableLlist, indiciL)
    result_list.extend(result_list1)
    result_list2, dataR = sampling_table(tableRlist, indiciR)
    result_list.extend(result_list2)
    return result_list, dataL, dataR, tableLlist, tableRlist


def sampling_table(table_list, indici):
    result_list1 = []
    data = []
    for j in range(len(table_list)):
        table_el = []
        for i1 in indici:
            table_el.append(table_list[j][i1])
        data.append(table_el)
        stringa_el = concatenate_list_data(table_el)
        lista_di_stringhe = stringa_el.split()
        result_list1.append(lista_di_stringhe)
    return result_list1, data


def minHash_LSH(data, threshold=0.65, num_perm=256, weights=(0.5, 0.5)):
    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, weights=weights)

    # Create MinHash objects
    minhashes = {}
    for c, i in enumerate(data):
        minhash = MinHash(num_perm=num_perm)
        for el in i:
            minhash.update(el.encode('utf8'))
        lsh.insert(c, minhash)
        minhashes[c] = minhash

    res_match = []
    for i in range(len(minhashes.keys())):
        result = lsh.query(minhashes[i])

        if result not in res_match and len(result) == 2:
            res_match.append(result)
    return res_match


def create_dataset_pt(res, data_l, data_r, table_l_list, table_r_list, min_sim, max_sim, dict_l_match, dict_r_match,
                      dict_l_no_match, dict_r_no_match, sim_function):
    dataPT = []
    i = 0
    for el in res:
        el1, table1, index1 = find_el(el[0], data_l, data_r)
        el2, table2, index2 = find_el(el[1], data_l, data_r)
        i = i + 1
        if table1 != table2:
            if table1 == "L":
                table_l_elem = concatenate_list_data(table_l_list[index1])
                table_r_elem = concatenate_list_data(table_r_list[index2])
            else:
                table_r_elem = concatenate_list_data(table_r_list[index1])
                table_l_elem = concatenate_list_data(table_l_list[index2])

            sim_vector = sim_function(el1, el2)
            if sim_vector[0] > max_sim:
                # match
                if count_occurrence(dict_l_match, table_l_elem) and count_occurrence(dict_r_match, table_r_elem):
                    dataPT.append((el1, el2, sim_vector))
            if sim_vector[0] < min_sim:
                # NO_match
                if count_occurrence(dict_l_no_match, table_l_elem) and count_occurrence(dict_r_no_match, table_r_elem):
                    dataPT.append((el1, el2, sim_vector))

    return dataPT


def find_el(index, data_l, data_r):
    if index >= len(data_l):
        ind_r = index - len(data_l)
        data_el = data_r[ind_r]
        table = "R"
        return data_el, table, ind_r
    else:
        data_el = data_l[index]
        table = "L"
        return data_el, table, index


def split_index(index):
    index_l = []
    index_r = []
    for i in range(len(index)):
        index_l.append(index[i][0])
        index_r.append(index[i][1])
    return index_l, index_r


def min_hash_lsh(table_l, table_r, index, min_sim, max_sim, dict_l_match, dict_r_match, dict_l_no_match, dict_r_no_match,
                 sim_function):
    index_l, index_r = split_index(index)
    data4hash, data_l, data_r, table_l_list, table_r_list = create_data(table_l, table_r, index_l, index_r)
    res = []
    weights = [0.4, 0.6]
    for threshold, num_perm in [(0.65, 256), (0.9, 64), (0.5, 128), (0.3, 64)]:
        res_c = minHash_LSH(data4hash, threshold=threshold, num_perm=num_perm, weights=weights)
        for r in res_c:
            if r not in res:
                res.append(r)
        logging.info("{} pairs found".format(len(res)))
    dataset_pt = create_dataset_pt(res, data_l, data_r, table_l_list, table_r_list, min_sim, max_sim, dict_l_match,
                                   dict_r_match, dict_l_no_match, dict_r_no_match, sim_function)
    logging.info("LSH blocking done")
    plot_data_pt(dataset_pt)

    return dataset_pt


def copy_edit_match(tupla):
    copy_tup = []

    for i in range(len(tupla)):
        change_attr = random.randint(0, 2)
        attr = Sequence(tupla[i])
        if len(tupla[i]) > 1 and change_attr == 1:
            d = 3  # max edit distance
            n = 3  # number of strings in result
            try:
                mutates = attr.mutate(d, n)
                # logging.info(mutates[1])

                copy_tup.append(str(mutates[1]))
            except:
                copy_tup.append(str(tupla[i]) + '  ')
        else:
            copy_tup.append(tupla[i])

    if copy_tup == tupla:
        copy_tup = copy_edit_match(tupla)
    return copy_tup


def create_flat_list(to_be_flatted):
    flat_list = []
    for el in to_be_flatted:
        table_elem = concatenate_list_data(el)
        flat_list.append(table_elem)
    return flat_list


def dict_tuple(csv_list):
    flat_list = create_flat_list(csv_list)
    dict_tuple = dict((el, 0) for el in flat_list)
    return dict_tuple


def count_occurrence(dictionary, tuple, limit='8'):
    if dictionary[tuple] < int(limit):
        dictionary[tuple] += 1
        return True
    else:
        return False


def csv_table2dataset_random_count_occ(table_l, table_r, total, min_sim, max_sim, index, min_cos_sim, tot_copy_match,
                                       max_occ, sim_function=lambda x, y: [1, 1]):
    loop_i = 0
    copy_match = 0
    table1 = csv.reader(open(table_l, encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(table_r, encoding="utf8"), delimiter=',')

    # skip header
    next(table1, None)
    next(table2, None)

    # convert to list for direct access
    table_l_list = list(table1)
    logging.info(len(table_l_list))
    table_r_list = list(table2)
    logging.info(len(table_r_list))

    dict_l_match = dict_tuple(table_l_list)
    logging.info(len(dict_l_match))
    dict_r_match = dict_tuple(table_r_list)
    logging.info(len(dict_r_match))
    dict_l_no_match = dict_tuple(table_l_list)
    logging.info(len(dict_l_no_match))
    dict_r_no_match = dict_tuple(table_r_list)
    logging.info(len(dict_r_no_match))

    result_list_no_match = []
    result_list_match = []
    copy_match_list = []

    logging.info("LSH blocking started")
    data_lsh = min_hash_lsh(table_l, table_r, index, min_sim, max_sim, dict_l_match, dict_r_match, dict_l_no_match,
                            dict_r_no_match, sim_function)
    for el in data_lsh:
        if el[2][0] >= max_sim and el not in result_list_match:
            result_list_match.append(el)

    logging.info(f'{len(result_list_match)} pairs found via LSH blocking and high similarity check')
    count_i = 0
    stop = False
    match = len(result_list_match)
    no_match = len(result_list_no_match)
    pair_max_visit = 10 * (total - match)
    logging.info(f'max pair visit: {pair_max_visit}')
    while loop_i < pair_max_visit and (match < total or no_match < total) and not stop:
        x = random.randint(1, len(table_l_list) - 1)
        y = random.randint(1, len(table_r_list) - 1)
        table_l_el = []
        table_r_el = []
        for i1, i2 in index:
            table_l_el.append(table_l_list[x][i1])
            table_r_el.append(table_r_list[y][i2])
        # to calculate cos_sim between the two elements of the tuple, requires concat the entire row
        string1 = concatenate_list_data(table_l_el)
        string2 = concatenate_list_data(table_r_el)
        cos_sim = get_cosine(text_to_vector(string1), text_to_vector(string2))

        # to count occurrence
        table_l_elem = concatenate_list_data(table_l_list[x])
        table_r_elem = concatenate_list_data(table_r_list[y])

        # the tuple I am adding needs to have cos_sim > min_cos_sim
        if cos_sim > min_cos_sim:
            sim_vector = sim_function(table_l_el, table_r_el)
            if sim_vector[0] > max_sim and match < total:
                if (table_l_el, table_r_el, sim_vector) not in result_list_match:
                    # match
                    if count_occurrence(dict_l_match, table_l_elem) and count_occurrence(dict_r_match, table_r_elem):
                        result_list_match.append((table_l_el, table_r_el, sim_vector))
                        marginal_entropy = (entropy1(concatenate_list_data(table_l_el).split(' ')) + entropy1(
                            concatenate_list_data(table_r_el).split(' '))) / (1 + len(result_list_match))
                        if marginal_entropy < 1e-4:
                            stop = True
                        match = match + 1

                        loop_i = 0
                    else:
                        loop_i = loop_i + 1
            elif sim_vector[0] < min_sim and no_match < (total + tot_copy_match):
                if (table_l_el, table_r_el, sim_vector) not in result_list_no_match:
                    # NO_match
                    if count_occurrence(dict_l_no_match, table_l_elem, limit=max_occ) and count_occurrence(dict_r_no_match,
                                                                                                        table_r_elem,
                                                                                                        limit=max_occ):
                        # NO_match
                        marginal_entropy = (entropy1(concatenate_list_data(table_l_el).split(' ')) + entropy1(
                            concatenate_list_data(table_r_el).split(' '))) / (1 + len(result_list_no_match))
                        if marginal_entropy < 1e-4:
                            stop = True
                        result_list_no_match.append((table_l_el, table_r_el, sim_vector))
                        no_match = no_match + 1
                        loop_i = 0
                    else:
                        loop_i = loop_i + 1

        elif copy_match < tot_copy_match:
            tableL_el2 = copy_edit_match(table_l_el)
            sim_vector = sim_function(table_l_el, tableL_el2)
            if (table_l_el, tableL_el2, sim_vector) not in result_list_match and sim_vector[0] > max_sim:
                # match
                if count_occurrence(dict_l_match, table_l_elem):
                    result_list_match.append((table_l_el, tableL_el2, sim_vector))
                    copy_match_list.append((table_l_el, tableL_el2, sim_vector))
                    copy_match = copy_match + 1

                    loop_i = 0

            tableR_el2 = copy_edit_match(table_r_el)
            sim_vector = sim_function(table_r_el, tableR_el2)
            if (table_r_el, tableR_el2, sim_vector) not in result_list_match and sim_vector[0] > max_sim:
                # match
                if count_occurrence(dict_r_match, table_r_elem, limit=max_occ):
                    result_list_match.append((table_r_el, tableR_el2, sim_vector))
                    copy_match_list.append((table_r_el, tableR_el2, sim_vector))
                    copy_match = copy_match + 1

                    loop_i = 0

        elif no_match < (total + tot_copy_match):
            sim_vector = sim_function(table_l_el, table_r_el)
            if sim_vector[0] < min_sim and (table_l_el, table_r_el, sim_vector) not in result_list_no_match:
                # NO_match
                if count_occurrence(dict_l_no_match, table_l_elem, limit=max_occ) and count_occurrence(dict_r_no_match,
                                                                                                    table_r_elem,
                                                                                                    limit=max_occ):
                    result_list_no_match.append((table_l_el, table_r_el, sim_vector))
                    no_match = no_match + 1
                    loop_i = 0
            else:
                loop_i = loop_i + 1
        else:
            loop_i = loop_i + 1
        count_i += 1

    logging.info("dizionari")
    plot_dicts(dict_l_match, dict_r_match, dict_l_no_match, dict_r_no_match)
    logging.info("create candidates set")
    return result_list_no_match, result_list_match


def create_lists(table_l, table_r, total, min_sim, max_sim, index, min_cos_sim, tot_copy_match,
                 max_occ, sim_function=lambda x, y: [1, 1]):
    # no duplicates

    loop_i = 0
    copies = 0
    table1 = csv.reader(open(table_l, encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(table_r, encoding="utf8"), delimiter=',')

    # skip header
    next(table1, None)
    next(table2, None)

    # convert to list for direct access
    table_llist = list(table1)
    logging.info(len(table_llist))
    table_rlist = list(table2)
    logging.info(len(table_rlist))

    # create dict for count the occorrence
    dict_l_match = dict_tuple(table_llist)

    logging.info(len(dict_l_match))
    dict_r_match = dict_tuple(table_rlist)

    logging.info(len(dict_r_match))
    dict_l_no_match = dict_tuple(table_llist)

    logging.info(len(dict_l_no_match))

    dict_r_no_match = dict_tuple(table_rlist)

    no_match = 0
    match = 0
    result_list_no_match = []
    result_list_match = []
    copies_list = []

    logging.info("LSH blocking started")
    data_lsh = min_hash_lsh(table_l, table_r, index, min_sim, max_sim, dict_l_match, dict_r_match, dict_l_no_match,
                            dict_r_no_match, sim_function)
    for el in data_lsh:
        if el[2][0] >= max_sim and el not in result_list_match:
            result_list_match.append(el)
        elif el[2][0] <= min_sim and el not in result_list_no_match:
            result_list_no_match.append(el)

    logging.info(f'{len(result_list_match)} positive pairs found via LSH blocking and high similarity check')
    logging.info(f'{len(result_list_no_match)} negative pairs found via LSH blocking and low similarity check')

    count_i = 0
    bigger_size = max(5 * total, 1000)
    logging.info(f'max pair visit: {bigger_size}')
    while loop_i < 120000 and count_i < bigger_size and (match < total or no_match < total):
        try:
            random.seed(count_i)
            x = random.randint(1, len(table_llist) - 1)
            y = random.randint(1, len(table_rlist) - 1)
            table_l_el = []
            table_r_el = []
            for i1, i2 in index:
                table_l_el.append(table_llist[x][i1])
                table_r_el.append(table_rlist[y][i2])
            # to calculate cos_sim between the two elements of the tuple, need to concatenate the entire row
            stringa1 = concatenate_list_data(table_l_el)
            stringa2 = concatenate_list_data(table_r_el)
            cos_sim = get_cosine(text_to_vector(stringa1), text_to_vector(stringa2))

            # to count occurrence
            table_l_elem = concatenate_list_data(table_llist[x])  # [ item for elem in table_llist[x] for item in elem]
            table_r_elem = concatenate_list_data(table_rlist[y])  # [ item for elem in table_rlist[y] for item in elem]

            # check tuple has cos_sim > min_cos_sim
            if cos_sim > 0:
                sim_vector = sim_function(table_l_el, table_r_el)
                if sim_vector[0] > max_sim and match < total:
                    if (table_l_el, table_r_el, sim_vector) not in result_list_match:
                        # match
                        if count_occurrence(dict_l_match, table_l_elem) and count_occurrence(dict_r_match, table_r_elem):
                            result_list_match.append((table_l_el, table_r_el, sim_vector))

                            match = match + 1

                            loop_i = 0
                        else:
                            loop_i = loop_i + 1
                elif sim_vector[0] < min_sim and no_match < (total + tot_copy_match):
                    if (table_l_el, table_r_el, sim_vector) not in result_list_no_match:
                        # NO_match
                        if count_occurrence(dict_l_no_match, table_l_elem, limit=max_occ) and count_occurrence(
                                dict_r_no_match,
                                table_r_elem,
                                limit=max_occ):
                            result_list_no_match.append((table_l_el, table_r_el, sim_vector))
                            no_match = no_match + 1
                            loop_i = 0

                        else:
                            loop_i = loop_i + 1

            elif copies < tot_copy_match:
                tableL_el2 = copy_edit_match(table_l_el)
                sim_vector = sim_function(table_l_el, tableL_el2)
                if (table_l_el, tableL_el2, sim_vector) not in result_list_match and (
                        sim_vector[0] > max_sim or sim_vector[0] < min_sim):
                    # match
                    if count_occurrence(dict_l_match, table_l_elem):

                        copies_list.append((table_l_el, tableL_el2, sim_vector))
                        copies = copies + 1

                        loop_i = 0

                tableR_el2 = copy_edit_match(table_r_el)
                sim_vector = sim_function(table_r_el, tableR_el2)
                if (table_r_el, tableR_el2, sim_vector) not in result_list_match and (
                        sim_vector[0] > max_sim or sim_vector[0] < min_sim):
                    # match
                    if count_occurrence(dict_r_match, table_r_elem, limit=max_occ):
                        copies_list.append((table_r_el, tableR_el2, sim_vector))
                        copies = copies + 1

                        loop_i = 0

            elif no_match < (total + tot_copy_match):
                sim_vector = sim_function(table_l_el, table_r_el)
                if sim_vector[0] < min_sim and (table_l_el, table_r_el, sim_vector) not in result_list_no_match:
                    # NO_match
                    if count_occurrence(dict_l_no_match, table_l_elem, limit=max_occ) and count_occurrence(dict_r_no_match,
                                                                                                        table_r_elem,
                                                                                                        limit=max_occ):
                        result_list_no_match.append((table_l_el, table_r_el, sim_vector))
                        no_match = no_match + 1
                        loop_i = 0
                else:
                    loop_i = loop_i + 1
            else:
                loop_i = loop_i + 1
            count_i += 1
        except Exception as e:
            print(traceback.format_exc())
            print(f'skipped item {str(count_i)}')

    logging.info("dizionari")
    plot_dicts(dict_l_match, dict_r_match, dict_l_no_match, dict_r_no_match)
    logging.info("create candidates set")
    return result_list_no_match, result_list_match, copies_list
