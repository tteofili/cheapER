import random
from collections import Counter

from cheaper.data.csv2dataset import csv_2_datasetALTERNATE
from cheaper.data.sampling_dataset_pt import csvTable2datasetRANDOM_countOcc
from cheaper.similarity.sim_function import sim4attrFZ
import logging
from cheaper.emt.logging_customized import setup_logging

setup_logging()

def count_occurrence_test(d, key):
    d[key] = d.get(key, 0) + 1


def count4attr(tuple,lista):
    for tupla in tuple:
        
        for i in range(len(tupla[0])):
            #logging.info(lista[i])
            count_occurrence_test(lista[i],tupla[0][i])
        
        for i in range(len(tupla[1])):
            #logging.info(lista[i])
            count_occurrence_test(lista[i],tupla[1][i])
            
    return lista

def init_dict_lista(lista_tupleMATCH,lista_tupleNO_MATCH,num_index):
    lista_attrMATCH=[]
    lista_attrNO_MATCH=[]
    for i in range(num_index):
        
        lista_attrMATCH.append({})
        lista_attrNO_MATCH.append({})
    #logging.info(lista_attrMATCH)
    lista_attrMATCH=count4attr(lista_tupleMATCH,lista_attrMATCH)
    lista_attrNO_MATCH=count4attr(lista_tupleNO_MATCH,lista_attrNO_MATCH)
    return lista_attrMATCH,lista_attrNO_MATCH
    
    
# TEST AREA #
if __name__ == "__main__":
    
    from plot import plotting, plot_dataPT, plot_pretrain, plotting_occorrenze

    #tableL='beer_exp_data/exp_data/tableA.csv'
    #tableR='beer_exp_data/exp_data/tableB.csv'
    
#    ground_truth='fodo_zaga//matches_fodors_zagats.csv'
#    TABLE1_FILE='fodo_zaga/fodors.csv'
#    TABLE2_FILE='fodo_zaga/zagats.csv'
#    ATT_INDEXES=[(1, 1), (2, 2), (3, 3),(4,4),(5,5),(6,6)]
        
    ground_truth='walmart_amazon/matches_walmart_amazon.csv'
    TABLE1_FILE='walmart_amazon/walmart.csv'
    TABLE2_FILE='walmart_amazon/amazonw.csv'
    ATT_INDEXES=[(5,9),(4,5),(3,3),(14,4),(6,11)]




    result_list=csv_2_datasetALTERNATE(ground_truth, TABLE1_FILE, TABLE2_FILE,ATT_INDEXES)
    
    min_sim=0.4
    max_sim=0.85
    min_cos_sim=0.2
    tot_copy=100
    
    simf = lambda a, b: sim4attrFZ(a, b)
    result_list_noMatch,result_list_match =csvTable2datasetRANDOM_countOcc(TABLE1_FILE,TABLE2_FILE,230,min_sim,max_sim,ATT_INDEXES,min_cos_sim, tot_copy, simf )
    
    lista_attrMATCH,lista_attrNO_MATCH=init_dict_lista(result_list_match,result_list_noMatch,len(ATT_INDEXES))
    logging.info("dizionari doppi degli attributi RANDOM")
    j=0
    for dictionario in lista_attrMATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrMATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            logging.info( '%s: %i' % (k, v))
    j=0
    for dictionario in lista_attrNO_MATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrNO_MATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            logging.info( '%s: %i' % (k, v))
            
    random_tuples0=result_list_noMatch+result_list_match
    random.shuffle(random_tuples0)
    random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
    logging.info("---------------- RANDOM TUPLES -------------------------")
    plot_pretrain(random_tuples0sort)
