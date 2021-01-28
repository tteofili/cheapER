# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:05:52 2020

@author: Giulia
"""
import csv
import random
from itertools import islice 
from sklearn.metrics.pairwise import cosine_similarity
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#### WARNING
import re, math
from collections import Counter


'''

def csv_2_datasetALTERNATE(ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1]):
    
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')
    matches_file = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
  
    #skip header
    next(table1, None)
    next(table2, None)
    next(matches_file, None)
    
    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    matches_list = list(matches_file)


    dictL_match=dict_tuple(tableLlist)
    #dictL_match=dict((el[0],0) for el in tableLlist)
    print(len(dictL_match))
    dictR_match=dict_tuple(tableRlist)
    print(len(dictR_match))
    #dictR_match=dict((el,0) for el in tableRlist)
    dictL_NOmatch=dict_tuple(tableLlist)
    print(len(dictL_NOmatch))
    #dictL_NOmatch=dict((el,0) for el in tableLlist)
    dictR_NOmatch=dict_tuple(tableRlist)
    print(len(dictR_NOmatch))
    
    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    #costruisce lista dei match parsando i file di input
    for line_in_file in matches_list:
        #line_in_file type: id_1, id_2
        row1=[item for item in tableLlist if item[0]==line_in_file[0]]
        row2=[item for item in tableRlist if item[0]==line_in_file[1]]
        
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(row1[0][i1])
            tableR_el.append(row2[0][i2])
        
        
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        
        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        cos_sim_list.append(cos_sim)
        
        sim_vector=sim_function(tableL_el,tableR_el) # Modificato
        
        #serve per il conta occorrenza
        tableL_ELEM = concatenate_list_data(row1[0]) #[ item for elem in tableLlist[x] for item in elem]
        tableR_ELEM = concatenate_list_data(row2[0]) #[ item for elem in tableRlist[y] for item in elem]
        
        
        count_occurrence_test(dictL_match, tableL_ELEM) 
        count_occurrence_test(dictR_match, tableR_ELEM)
        result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
        #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
        min_cos_sim_match=min(cos_sim_list)
    

##[1:] serve per togliere l id come attributo
    #costruisce la lista dei NO_match calcolando un min cos similarity 
    i=0
    while i<len(result_list_match):

        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
                
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])              
        
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        
        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)
            
            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
                tableL_ELEM = concatenate_list_data(tableLlist[x]) #[ item for elem in tableLlist[x] for item in elem]
                tableR_ELEM = concatenate_list_data(tableRlist[y]) #
                if count_occurrence(dictL_NOmatch, tableL_ELEM,10) and count_occurrence(dictR_NOmatch, tableR_ELEM,10):
                    count_occurrence_test(dictL_NOmatch, tableL_ELEM) 
                    count_occurrence_test(dictR_NOmatch, tableR_ELEM)
                    result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                    i += 1
    print(result_list_match[0])
    lista_OCCUREnce=[dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch]
    for dictionario in lista_OCCUREnce:
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            print( '%s: %i' % (k, v))
            
    lista_attrMATCH,lista_attrNO_MATCH=init_dict_lista(result_list_match,result_list_NOmatch,len(indici))
    plotting_dizionari(dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch)
    print("dizionari doppi degli attributi")
    j=0
    for dictionario in lista_attrMATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrMATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            print( '%s: %i' % (k, v))
    j=0
    for dictionario in lista_attrNO_MATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrNO_MATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            print( '%s: %i' % (k, v))
    #list(lista_attrMATCH.values()
    #list(lista_attrNO_MATCH.values()
    #unisce le due liste dei match e No_match alternandole
    result_list=[]
    #random.shuffle(result_list_match)
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list
'''
def count_occurrence_test(d, key):
    d[key] = d.get(key, 0) + 1


def count4attr(tuple,lista):
    for tupla in tuple:
        
        for i in range(len(tupla[0])):
            #print(lista[i])
            count_occurrence_test(lista[i],tupla[0][i])
        
        for i in range(len(tupla[1])):
            #print(lista[i])
            count_occurrence_test(lista[i],tupla[1][i])
            
    return lista

def init_dict_lista(lista_tupleMATCH,lista_tupleNO_MATCH,num_index):
    lista_attrMATCH=[]
    lista_attrNO_MATCH=[]
    for i in range(num_index):
        
        lista_attrMATCH.append({})
        lista_attrNO_MATCH.append({})
    #print(lista_attrMATCH)
    lista_attrMATCH=count4attr(lista_tupleMATCH,lista_attrMATCH)
    lista_attrNO_MATCH=count4attr(lista_tupleNO_MATCH,lista_attrNO_MATCH)
    return lista_attrMATCH,lista_attrNO_MATCH
    
    
# TEST AREA #
if __name__ == "__main__":
    
    from sim_function import sim_cos, sim4attrFZ#,sim4attrFZ_norm,sim4attrFZ_norm2
    from plot import plotting,plot_dataPT,plot_pretrain 
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
    print("dizionari doppi degli attributi RANDOM")
    j=0
    for dictionario in lista_attrMATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrMATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            print( '%s: %i' % (k, v))
    j=0
    for dictionario in lista_attrNO_MATCH:
        j=j+1
        plotting_occorrenze(list(dictionario.values()), f'lista_attrNO_MATCH{j}')
        d = Counter(dictionario)
        for k, v in d.most_common(5):
            print( '%s: %i' % (k, v))
            
    random_tuples0=result_list_noMatch+result_list_match
    random.shuffle(random_tuples0)
    random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
    print("---------------- RANDOM TUPLES -------------------------")
    plot_pretrain(random_tuples0sort)

#lista_attrMATCH,lista_attrNO_MATCH=init_dict_lista(lista_tuple,len(ATT_INDEXES))

#lista_attr_dict=[dict_attr1M,dict_attr2M,dict_attr3M,dict_attr4M]
#lista_attr_dict=count4attr(lista_tuple,lista_attr_dict)#dict_attr1M,dict_attr2M,dict_attr3M,dict_attr4M)
#lista=count4attr(tuplaA,lista)

      
