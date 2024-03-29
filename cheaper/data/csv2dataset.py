import csv
import logging
#### WARNING
import math
import random
import re
from collections import Counter
from itertools import islice

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cheaper.data.edit_dna import Sequence
from cheaper.data.plot import plotting_dizionari
from cheaper.emt.logging_customized import setup_logging
from cheaper.similarity.sim_function import sim4attrFZ_norm2

setup_logging()
'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!IMPORTANTE!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!IMPORTANTE!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''

WORD = re.compile(r'\w+')

#calcola la cos similarity di due vettori
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def concatenate_list_data(list):
    result= ''
    for element in list:
        result += ' '+str(element)
    return result

##secondo metodo per il calcolo del cos-sim NON UTILIZZATO
def cos_sim2Str(str1,str2):
    documents=[str1,str2]
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=None)
    sparse_matrix = count_vectorizer.fit_transform(documents)
    # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,columns=count_vectorizer.get_feature_names(),index=['str1', 'str2'])
    #logging.info(df)
    cos_sim=cosine_similarity(df,df)
    #logging.info(cos_sim[0][-1])
    return cos_sim[0][-1]

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''parsing del csv e costruzione dataset => (tupla1, tupla2, vettore_sim, label_match_OR_no_match)
    con shuffle finale dei match-No match'''
def csv_2_dataset(ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1]):

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

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]

    #costruisce lista dei match
    for line_in_file in matches_list:
        #line_in_file type: id_1, id_2
        row1=[item for item in tableLlist if item[0]==line_in_file[0]]
        row2=[item for item in tableRlist if item[0]==line_in_file[1]]

        #logging.info(row1[0])
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(row1[0][i1])
            tableR_el.append(row2[0][i2])


        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        #calcola la cos_sim tra le due righe
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #logging.info(cos_sim)
        cos_sim_list.append(cos_sim)
        #calcola il vettore di similarita
        sim_vector=sim_function(tableL_el,tableR_el) # Modificato

        result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
        #minimo valore di cos_similarità tra tutte le tuple in match
        min_cos_sim_match=min(cos_sim_list)


##[1:] serve per togliere l id come attributo

    #costruzione lista dei non match
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
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match e che non sia nella lista dei match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :

                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i += 1

    #unisce le due liste (match e non match) e fa uno shuffle
    result_list_match.extend(result_list_NOmatch)
    random.shuffle(result_list_match)

    return result_list_match

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''parsing dei csv e costruzione dataset => (tupla1,tupla2,vettore_sim) completamente random 
    data una lunghezza totale (tot)
    indici= lista di Coppie di attributi considerati   (es: per Walmart Amazon (Walmart_att, Amazon_att))
    cosi ogni coppia di tuple ha stesso num di attributi
    ES:   indici=[(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]
    si utilizza una soglia di cos similarity definita STATICAMENTE per inserire la tupla
    (se non si vuole utilizzare basta imporla a zero)'''

def csvTable2datasetRANDOM(tableL, tableR, tot, indici, sim_function=lambda x, y: [1, 1]):
    '''#soglia di cos similarità '''
    min_cos_sim=0.17

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)

    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
    while i<tot:
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

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            sim_vector=sim_function(tableL_el,tableR_el)
            result_list.append((tableL_el,tableR_el,sim_vector))
            i += 1


    return result_list

def csvTable2datasetRANDOMCos(tableL, tableR, tot, indici, sim_function=lambda x, y: [1, 1]):
    '''#soglia di cos similarità '''
    min_cos_sim=0.17

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)

    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
    while i<tot:
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

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            sim_vector=[cos_sim]#sim_function(tableL_el,tableR_el)
            result_list.append((tableL_el,tableR_el,sim_vector))
            i += 1


    return result_list

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''parsing dei csv e costruzione dataset => (tupla1,tupla2,vettore_sim) completamente random 
    data una lunghezza totale (tot), 
    min_cos_sim= una soglia di cos_similarità minima,
    indici= lista di Coppie di attributi considerati   (es: per Walmart Amazon (Walmart_att, Amazon_att))
    cosi ogni coppia di tuple ha stesso num di attributi
    ES:   indici=[(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]'''

def csvTable2datasetRANDOM_minCos(tableL, tableR, tot, indici, min_cos_sim, sim_function=lambda x, y: [1, 1]):

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)

    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
    while i<tot:
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

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            sim_vector=sim_function(tableL_el,tableR_el)
            result_list.append((tableL_el,tableR_el,sim_vector))
            i += 1
    #logging.info(result_list)
    return result_list


def csvTable2datasetRANDOM_bilanced_checkAllList(tableL, tableR, tot, indici, sim_function=lambda x, y: [1, 1]):
    '''#soglia di cos similarità '''
    min_cos_sim=0

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
    for x in range(len(tableLlist)):
        for y in range(len(tableRlist)):
            while i<tot:
                #x = random.randint(1,len(tableLlist)-1)
                #y =  random.randint(1,len(tableRlist)-1)
                tableL_el=[]
                tableR_el=[]
                for i1,i2 in indici:
                    tableL_el.append(tableLlist[x][i1])
                    tableR_el.append(tableRlist[y][i2])
                sim_vector=sim_function(tableL_el,tableR_el)
                if sim_vector[0]>0.5 and match<tot/2:
                    result_list.append((tableL_el,tableR_el,sim_vector))
                    match=match+1
                    i=i+1
                    logging.info("lista random match: " +str(match))

                if sim_vector[0]<0.2 and no_match<tot/2:
                    result_list.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    i=i+1
                    logging.info("lista random no match: " +str(no_match))

    logging.info(result_list[:4])
    return result_list

def csvTable2datasetRANDOM_bilanced1(tableL, tableR, tot,min_sim,max_sim, indici, sim_function=lambda x, y: [1, 1] ):
    '''#soglia di cos similarità '''
    min_cos_sim=0

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
#    for x in range(len(tableLlist)):
#        for y in range(len(tableRlist)):
    while i<tot:
        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])
        sim_vector=sim_function(tableL_el,tableR_el)
        if sim_vector[0]>max_sim and match<tot/2:
            #if (tableL_el,tableR_el,sim_vector) not in result_list:

            result_list.append((tableL_el,tableR_el,sim_vector))
            match=match+1
            i=i+1
            logging.info("lista random match: " +str(match))

        if sim_vector[0]<min_sim and no_match<tot/2:
            #if (tableL_el,tableR_el,sim_vector) not in result_list:

            result_list.append((tableL_el,tableR_el,sim_vector))
            no_match=no_match+1
            i=i+1
            logging.info("lista random no match: " +str(no_match))

    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list

def csvTable2datasetRANDOM_bilanced(tableL, tableR, tot,min_sim,max_sim, indici, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''
    min_cos_sim=0

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
#    for x in range(len(tableLlist)):
#        for y in range(len(tableRlist)):
    while i<tot:
        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])
        sim_vector=sim_function(tableL_el,tableR_el)
        if sim_vector[0]>max_sim and match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                match=match+1
                i=i+1
                logging.info("lista random match: " +str(match))

        if sim_vector[0]<min_sim and no_match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                i=i+1
                logging.info("lista random no match: " +str(no_match))

    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list
#def create_match(tableL, tableR, totale,min_sim,max_sim, indici, sim_function=lambda x, y: [1, 1]):


def csvTable2datasetRANDOM_bilancedWITHlsh(tableL, tableR, totale,min_sim,max_sim, indici,data_lsh, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''
    min_cos_sim=0
    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    #totale=3000
    ##[1:] serve per togliere l id come attributo
    i=0
#    for x in range(len(tableLlist)):
#        for y in range(len(tableRlist)):
    for el in data_lsh:
            if el[2][0]>max_sim and el not in result_list:
                result_list.append(el)
                match=match+1
                logging.info("lista lsh match: " +str(match))
            if el[2][0]<min_sim and el not in result_list:
                result_list.append(el)
                no_match=no_match+1
                logging.info("lista lsh no match: " +str(no_match))

    while loop_i<300000:# and (match<totale or no_match<totale):
        #if loop_i%1000000==0:
            #logging.info("loop_i: "+str(loop_i))


        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])
        sim_vector=sim_function(tableL_el,tableR_el)
        if sim_vector[0]>max_sim and match<6000:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                match=match+1

                logging.info("lista random match: " +str(match)+" loop_i: "+str(loop_i))
                loop_i=0
            else:
                loop_i=loop_i+1
        elif sim_vector[0]<min_sim and no_match<totale:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                loop_i=0
                logging.info("lista random no match: " +str(no_match)+" loop_i: "+str(loop_i))
            else:
                loop_i=loop_i+1
        elif copy_match<50:
            sim_vector=sim_function(tableL_el,tableL_el)
            if (tableL_el,tableL_el,sim_vector) not in result_list:
                result_list.append((tableL_el,tableL_el,sim_vector))
                copy_match=copy_match+1

                logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                loop_i=0
            else:
                sim_vector=sim_function(tableR_el,tableR_el)
                if (tableR_el,tableR_el,sim_vector) not in result_list:
                    result_list.append((tableR_el,tableR_el,sim_vector))
                    copy_match=copy_match+1

                    logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    loop_i=0

        else:
            loop_i=loop_i+1
    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list



def csvTable2datasetRANDOM_nomatch(tableL, tableR, tot,min_sim, indici, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''
    min_cos_sim=0

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
#    for x in range(len(tableLlist)):
#        for y in range(len(tableRlist)):
    while i<tot:
        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])
        sim_vector=sim_function(tableL_el,tableR_el)
        '''
        if sim_vector[0]>max_sim and match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:
                
                result_list.append((tableL_el,tableR_el,sim_vector))
                match=match+1
                i=i+1
                logging.info("lista random match: " +str(match))
        '''
        if sim_vector[0]<min_sim and no_match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                i=i+1
                logging.info("lista random no match: " +str(no_match))

    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list
def csvTable2datasetRANDOM_extreme(tableL, tableR, indici, sim_function=lambda x, y: [1, 1] ):


    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []


    num_randomL=10*(math.log(len(tableLlist)))
    logging.info(num_randomL)
    num_randomR=10*(math.log(len(tableRlist)))
    logging.info(num_randomR)

    logging.info("parsing table R")
    for i in range(len(tableLlist)):
        if i%50==0:
            logging.info(i)
        j=0
        while j<num_randomR:
            #logging.info(j)
            y =  random.randint(1,len(tableRlist)-1)
            tableL_el=[]
            tableR_el=[]
            for i1,i2 in indici:
                tableL_el.append(tableLlist[i][i1])
                tableR_el.append(tableRlist[y][i2])
            sim_vector=sim_function(tableL_el,tableR_el)
            if (tableL_el,tableR_el,sim_vector) not in result_list:
                if sim_vector[0]>0.6:
                    match=match+1
                if sim_vector[0]<0.2:
                    no_match=no_match+1
                #logging.info(sim_vector[0])
                result_list.append((tableL_el,tableR_el,sim_vector))
            j=j+1
    logging.info("parsing table L")
    for k in range(len(tableRlist)):
        if k%50==0:
            logging.info(k)
        j=0
        while j<num_randomL:
            x = random.randint(1,len(tableLlist)-1)
            tableL_el=[]
            tableR_el=[]
            for i1,i2 in indici:
                tableL_el.append(tableLlist[x][i1])
                tableR_el.append(tableRlist[k][i2])
            sim_vector=sim_function(tableL_el,tableR_el)
            if (tableL_el,tableR_el,sim_vector) not in result_list:
                if sim_vector[0]>0.6:
                    match=match+1
                if sim_vector[0]<0.2:
                    no_match=no_match+1
                #logging.info(sim_vector[0])
                result_list.append((tableL_el,tableR_el,sim_vector))
            j=j+1

    logging.info(result_list[:4])
    logging.info(result_list[-4:])
    return result_list,match,no_match,num_randomL,num_randomR


def csvTable2datasetRANDOM_extremeRIP(tableL, tableR, indici,ripA,ripB, sim_function=lambda x, y: [1, 1] ):


    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []


    num_randomL=ripA*(math.log(len(tableLlist)))
    logging.info("num_randomL "+str(num_randomL))
    num_randomR=ripB*(math.log(len(tableRlist)))
    logging.info("num_randomR "+str(num_randomR))

    logging.info("parsing table R")
    for i in range(len(tableLlist)):
        if i%50==0:
            logging.info(str(i)+"/"+str(len(tableLlist)))
        j=0
        while j<num_randomR:
            #logging.info(j)
            y =  random.randint(1,len(tableRlist)-1)
            tableL_el=[]
            tableR_el=[]
            for i1,i2 in indici:
                tableL_el.append(tableLlist[i][i1])
                tableR_el.append(tableRlist[y][i2])
            sim_vector=sim_function(tableL_el,tableR_el)
            if (tableL_el,tableR_el,sim_vector) not in result_list:
                if sim_vector[0]>0.8:
                    match=match+1
                if sim_vector[0]<0.2:
                    no_match=no_match+1
                #logging.info(sim_vector[0])
                result_list.append((tableL_el,tableR_el,sim_vector))
            j=j+1
    logging.info("parsing table L")
    for k in range(len(tableRlist)):
        if k%50==0:
            logging.info(str(k)+"/"+str(len(tableRlist)))
        j=0
        while j<num_randomL:
            x = random.randint(1,len(tableLlist)-1)
            tableL_el=[]
            tableR_el=[]
            for i1,i2 in indici:
                tableL_el.append(tableLlist[x][i1])
                tableR_el.append(tableRlist[k][i2])
            sim_vector=sim_function(tableL_el,tableR_el)
            if (tableL_el,tableR_el,sim_vector) not in result_list:
                if sim_vector[0]>0.8:
                    match=match+1
                if sim_vector[0]<0.2:
                    no_match=no_match+1
                #logging.info(sim_vector[0])
                result_list.append((tableL_el,tableR_el,sim_vector))
            j=j+1

    logging.info(result_list[:4])
    logging.info(result_list[-4:])
    return result_list,match,no_match,num_randomL,num_randomR
"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''parsing dei csv e costruzione dataset alternato(match-NOmatch) =>  (tupla1,tupla2,vettore_sim,label_match_OR_no_match)
    indici= lista di Coppie di attributi considerati   (es: per Walmart Amazon (Walmart_att, Amazon_att))
    cosi ogni coppia di tuple ha stesso num di attributi
    ES:   indici=[(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]'''

def csv_2_datasetALTERNATE(ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1], max_len=-1, cut=1):

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
    if cut < 1:
        sl = int(max(len(matches_list)*cut, 5))
        matches_list = matches_list[:sl]

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
    for line_in_file in matches_list:
        #line_in_file type: id_1, id_2
        try:
            if line_in_file[2] == 0:
                continue

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
            #logging.info(cos_sim)
            cos_sim_list.append(cos_sim)

            sim_vector=sim_function(tableL_el,tableR_el) # Modificato

            result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
            #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
            min_cos_sim_match=min(cos_sim_list)
            if max_len > 0:
                if len(result_list_match) > max_len:
                    break
        except:
            pass

    ##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
    i=0
    while i<len(result_list_match):

        if result_list_match[i][3] == 0:
            break

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
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim >= min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :

                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i += 1

    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    #random.shuffle(result_list_match)
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list

def csv_2_datasetALTERNATEcos(ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1]):

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

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
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
        #logging.info(cos_sim)
        cos_sim_list.append(cos_sim)

        sim_vector=[cos_sim]#sim_function(tableL_el,tableR_el) # Modificato

        result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
        #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
        min_cos_sim_match=min(cos_sim_list)


##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
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
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=[cos_sim]#sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :

                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i += 1

    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list

def parsing_anhai_dataOnlyMatch(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):

    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_match=[]
    result_list_NOmatch=[]
    result_list=[]

    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        try:
            if int(line_in_file[2])==1:

                row1=[item for item in tableAlist if item[0]==line_in_file[0]]
                row2=[item for item in tableBlist if item[0]==line_in_file[1]]
                tableA_el=[]
                tableB_el=[]
                for i1,i2 in indici:
                    tableA_el.append(row1[0][i1])
                    tableB_el.append(row2[0][i2])


                stringaA=concatenate_list_data(tableA_el)
                stringaB=concatenate_list_data(tableB_el)

                #calcola la cos similarita della tupla i-esima
                cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
                #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
                #logging.info(cos_sim)
                sim_vector=sim_function(tableA_el,tableB_el) # Modificato
                cos_sim_list.append(cos_sim)
                result_list_match.append((tableA_el,tableB_el,sim_vector, 1))
        except:
            pass

    min_cos_sim_match=min(cos_sim_list)
    logging.info("min_cos_sim_match"+str(min_cos_sim_match))


##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
    i=0
    while i<len(result_list_match):

        x = random.randint(1,len(tableAlist)-1)
        y =  random.randint(1,len(tableBlist)-1)
        tableL_el=[]
        tableR_el=[]

        for i1,i2 in indici:
            tableL_el.append(tableAlist[x][i1])
            tableR_el.append(tableBlist[y][i2])

        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :

                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i=i+1

        '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list

def parsing_anhai_dataOnlyMatchCos(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):

    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_match=[]
    result_list_NOmatch=[]
    result_list=[]

    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        if int(line_in_file[2])==1:

            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])


            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)

            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #logging.info(cos_sim)
            sim_vector=[cos_sim]#sim_function(tableA_el,tableB_el) # Modificato
            cos_sim_list.append(cos_sim)
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))
            if (len(result_list_match)%10)==0:
                logging.info(len(result_list_match))

    min_cos_sim_match=min(cos_sim_list)
    logging.info("min_cos_sim_match"+str(min_cos_sim_match))


##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
    i=0
    while i<len(result_list_match):

        x = random.randint(1,len(tableAlist)-1)
        y =  random.randint(1,len(tableBlist)-1)
        tableL_el=[]
        tableR_el=[]

        for i1,i2 in indici:
            tableL_el.append(tableAlist[x][i1])
            tableR_el.append(tableBlist[y][i2])

        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=[cos_sim]#sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :

                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i=i+1

        '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list

def parsing_anhai_dataOnlyMatch4roundtraining(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):

    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_match=[]
    result_list_NOmatch=[]
    result_list=[]

    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        if int(line_in_file[2])==1:

            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])


            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)

            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #logging.info(cos_sim)
            sim_vector=sim_function(tableA_el,tableB_el) # Modificato
            cos_sim_list.append(cos_sim)
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))


    min_cos_sim_match=min(cos_sim_list)
    logging.info("min_cos_sim_match"+str(min_cos_sim_match))


##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
    i=0
    sim_list=[]
    while i<len(result_list_match):
        x = random.randint(1,len(tableAlist)-1)
        y =  random.randint(1,len(tableBlist)-1)
        tableL_el=[]
        tableR_el=[]

        for i1,i2 in indici:
            tableL_el.append(tableAlist[x][i1])
            tableR_el.append(tableBlist[y][i2])

        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
                sim_list.append(sim_vector)
                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i=i+1

        '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])
    average=max(sim_list)

    return result_list,average[0]


def parsing_anhai_data(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):

    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_NOmatch=[]
    result_list_match=[]

    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2

        row1=[item for item in tableAlist if item[0]==line_in_file[0]]
        row2=[item for item in tableBlist if item[0]==line_in_file[1]]
        tableA_el=[]
        tableB_el=[]
        for i1,i2 in indici:
            tableA_el.append(row1[0][i1])
            tableB_el.append(row2[0][i2])


        stringaA=concatenate_list_data(tableA_el)
        stringaB=concatenate_list_data(tableB_el)

        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #logging.info(cos_sim)
        sim_vector=sim_function(tableA_el,tableB_el) # Modificato
        if int(line_in_file[2])==1:
            cos_sim_list.append(cos_sim)
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))
        else:
            result_list_NOmatch.append((tableA_el,tableB_el,sim_vector, 0))


    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    #random.shuffle(result_list_match)
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    i=0
    flag=True
    while (i<len(result_list) and flag):
        el=result_list[i][3]
        if int(el)==1:
            first_match=result_list.pop(i)
            result_list.insert(0, first_match)
            flag=False
        else:
            i=i+1

    return result_list

def ratio_dupl_noDup4Anhai(dataset,index):
    sort_dataset = sorted(dataset, key=lambda tup: (tup[3], tup[2][index]))
    match_number=sum(map(lambda x : x[3] == 1, sort_dataset))
    logging.info("match_number: "+str(match_number))
    n=len(sort_dataset)-(match_number*2)
    logging.info(n)
    listout=sort_dataset[n:]
#    for i in range(105,115):
#        logging.info(listout[i])
    return listout


def check_anhai_dataset(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1], cut=1):

    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    No_match_with_cos_too_small=0
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)

    if cut < 1:
        sl = int(max(len(trainFilelist)*cut, 5))
    else:
        sl = len(trainFilelist)

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2

        if int(line_in_file[2])==1 and len(result_list_match) < sl:#se è un match
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])


            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)

            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #logging.info(cos_sim)
            cos_sim_list.append(cos_sim)

            sim_vector=sim_function(tableA_el,tableB_el) # Modificato

            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))
            #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
    min_cos_sim_match=min(cos_sim_list)
    logging.info("min coseno match:"+str(min_cos_sim_match))

    for line_in_file in trainFilelist:
        #else:
        if int(line_in_file[2])==0 and len(result_list_NOmatch) < sl:
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]

            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])


            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)

            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #logging.info(cos_sim)
            #cos_sim_list.append(cos_sim)
            if cos_sim<min_cos_sim_match:
                No_match_with_cos_too_small=No_match_with_cos_too_small+1
            else:
                sim_vector=sim_function(tableA_el,tableB_el) # Modificato

                result_list_NOmatch.append((tableA_el,tableB_el,sim_vector, 0))


    logging.info(max(len(result_list_match),len(result_list_NOmatch)))
    logging.info(len(result_list_match))
    logging.info(len(result_list_NOmatch))

    logging.info("match_tuple: "+str(len(result_list_match)))
    logging.info("no match_tuple: "+str(len(result_list_NOmatch)))
    logging.info("No_match_with_cos_too_small: "+str(No_match_with_cos_too_small))

    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    random.shuffle(result_list_NOmatch)
    i = 0
    j = 0
    while i < len(result_list_match) or j < len(result_list_NOmatch):
        if i < len(result_list_match):
            result_list.append(result_list_match[i])
        if j < len(result_list_NOmatch):
            result_list.append(result_list_NOmatch[j])
        i += 1
        j += 1

    return result_list
''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''

# Ti restituisce i primi n valori di datatasetOriginal
# con n = len(datasetOriginal) * percentuale
def splitting_dataSet(percentuale, dataSetOriginal):
    lunghezza=int(len(dataSetOriginal)*percentuale)
    "Return first n items of the iterable as a list"
    output=list(islice(dataSetOriginal, lunghezza))

    #logging.info("Split length list: ", percentuale)
    #logging.info("List after splitting", output)
    return output


'''caso label 0/1'''
def splitDataSet01WithPercent(percent, dataSetOriginal, percent1, percent0):
    logging.info(dataSetOriginal)
    lunghezza=int(round(len(dataSetOriginal)*percent))
    percentuale1=int(lunghezza*percent1)
    #logging.info(percentuale1)
    percentuale0=int(lunghezza*percent0)
    #logging.info(percentuale0)
    if int(percentuale1+percentuale0)!=lunghezza:
        percentuale1=percentuale1+1
    output=[]
    i=0
    logging.info('percentuale 1= '+str(percentuale1))
    logging.info('percentuale 0= '+str(percentuale0))

    while i<lunghezza:

        for elem in dataSetOriginal:
            if int(elem[2])==1 and percentuale1!=0:
                output.append(elem)
                percentuale1=percentuale1-1
                i=i+1
            elif int(elem[2])==0 and percentuale0!=0:
                    output.append(elem)
                    percentuale0=percentuale0-1
                    i=i+1

    #logging.info(output)
    return output # Modificato


'''caso vector similarity'''
def splitDataSetSIMWithPercent(percent, dataSetOriginal, percent1, percent0):
    lunghezza=int(round(len(dataSetOriginal)*percent))
    percentuale1=int(lunghezza*percent1)
    #logging.info(percentuale1)
    percentuale0=int(lunghezza*percent0)
    #logging.info(percentuale0)
    if int(percentuale1+percentuale0)!=lunghezza:
        percentuale1=percentuale1+1
    output=[]
    i=0
    logging.info('percentuale 1= '+str(percentuale1))
    logging.info('percentuale 0= '+str(percentuale0))
    while i<lunghezza:
        for elem in dataSetOriginal:
            if int(elem[3])==1 and percentuale1!=0:
                output.append(elem)
                percentuale1=percentuale1-1
                i=i+1
            elif int(elem[3])==0 and percentuale0!=0:
                    output.append(elem)
                    percentuale0=percentuale0-1
                    i=i+1

    #logging.info(output)
    return output # Modificato

def check_min_max(sim_vector,min_sim,max_sim):
    if not (min_sim and max_sim):
        min_sim=sim_vector.copy()
        max_sim=sim_vector.copy()
    else:
        for i in range(len(sim_vector)):
            min_sim[i]=min(min_sim[i],sim_vector[i])
            max_sim[i]=max(max_sim[i],sim_vector[i])
    return min_sim,max_sim

def csv_2_datasetALTERNATE_NORM(ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1]):

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

    sim_list=[]

    match_tempList=[]
    no_match_tempList=[]
    min_sim=[]
    max_sim=[]

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
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
        #logging.info(cos_sim)
        cos_sim_list.append(cos_sim)

        sim_vector,sim_attr=sim_function(tableL_el,tableR_el) # Modificato

        min_sim,max_sim=check_min_max(sim_attr,min_sim,max_sim)

        sim_list.append(sim_vector[0])
        match_tempList.append((tableL_el,tableR_el))
        result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
        #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
        min_cos_sim_match=min(cos_sim_list)


##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''
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
        #logging.info(cos_sim)

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector,sim_attr=sim_function(tableL_el,tableR_el)
            min_sim,max_sim=check_min_max(sim_attr,min_sim,max_sim)

            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
                sim_list.append(sim_vector[0])
                no_match_tempList.append((tableL_el,tableR_el))
                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i += 1

    '''unisce le due liste dei match e No_match alternandole'''
    result_list_norm=[]
    for i in range(max(len(match_tempList),len(no_match_tempList))):

        sim_value=sim4attrFZ_norm2(match_tempList[i][0],match_tempList[i][1],min_sim,max_sim)

        result_list_norm.append((match_tempList[i][0],match_tempList[i][1],sim_value, 1))
        sim_value=sim4attrFZ_norm2(no_match_tempList[i][0],no_match_tempList[i][1],min_sim,max_sim)
        result_list_norm.append((no_match_tempList[i][0],no_match_tempList[i][1],sim_value, 0))

    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])

    logging.info("min_sim")
    logging.info(min_sim)
    logging.info("max_sim")
    logging.info(max_sim)

    #return result_list_norm,match_tempList,no_match_tempList,result_list,sim_list,min_sim,max_sim

    return result_list_norm,min_sim,max_sim



def csvTable2datasetRANDOM_bil_NORM(tableL, tableR, tot,min_sim,max_sim, indici,min_simVect,max_simVect, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''
    min_cos_sim=0

    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    ##[1:] serve per togliere l id come attributo
    i=0
#    for x in range(len(tableLlist)):
#        for y in range(len(tableRlist)):
    while i<tot:
        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])
        sim_vector=sim4attrFZ_norm2(tableL_el,tableR_el,min_simVect,max_simVect)
        #sim_vector=sim_function(tableL_el,tableR_el)
        if sim_vector[0]>max_sim and match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                match=match+1
                i=i+1
                logging.info("lista random match: " +str(match))

        if sim_vector[0]<min_sim and no_match<tot/2:
            if (tableL_el,tableR_el,sim_vector) not in result_list:

                result_list.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                i=i+1
                logging.info("lista random no match: " +str(no_match))

    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list


def copy_EDIT_match(tupla):
    copy_tup=[]

    for i in range(len(tupla)):
        change_attr=random.randint(0,2)
        attr=Sequence(tupla[i])
        if len(tupla[i])>1 and change_attr==1:
            #logging.info(attr)
            d = 1  # max edit distance
            n = 3  # number of strings in result
            mutates=attr.mutate(d, n)
            #logging.info(mutates[1])

            copy_tup.append(str(mutates[1]))
        else:
            copy_tup.append(tupla[i])

    #logging.info(copy_tup)
    return copy_tup



def create_flat_list(lista):
    flatList = []
    for el in lista:
        tableELEM = concatenate_list_data(el)
        flatList.append(tableELEM)
    return flatList

def dict_tuple(csv_list):
    flatList=create_flat_list(csv_list)
    logging.info("elemento 0 della flat list")
    logging.info(flatList[0])
    dictTuple=dict((el,0) for el in flatList)
    return dictTuple

def csvTable2datasetRANDOM_likeGold(tableL,tableR,totale,min_sim,max_sim,indici,data_lsh,min_cos_sim,tot_copy_match, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''

    def count_occurrence(dizion, tupla):
        dizion[tupla] += 1


    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    logging.info(len(tableLlist))
    tableRlist = list(table2)
    logging.info(len(tableRlist))
    logging.info("sono il csv riga 0")
    logging.info(tableLlist[0])
    #create dict for count the occorrence


    dictL_match=dict_tuple(tableLlist)
    #dictL_match=dict((el[0],0) for el in tableLlist)
    logging.info(len(dictL_match))
    dictR_match=dict_tuple(tableRlist)
    logging.info(len(dictR_match))
    #dictR_match=dict((el,0) for el in tableRlist)
    dictL_NOmatch=dict_tuple(tableLlist)
    logging.info(len(dictL_NOmatch))
    #dictL_NOmatch=dict((el,0) for el in tableLlist)
    dictR_NOmatch=dict_tuple(tableRlist)
    logging.info(len(dictR_NOmatch))
    #dictR_NOmatch=dict((el,0) for el in tableRlist)

    no_match=0
    match=0
    result_list= []
    copy_match_list=[]
    for el in data_lsh:
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(el[0])
        stringa2=concatenate_list_data(el[1])
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            if el[2][0]>max_sim and el not in result_list:
                result_list.append(el)

                match=match+1
                logging.info("lista lsh match: " +str(match))
            if el[2][0]<min_sim and el not in result_list:
                result_list.append(el)
                no_match=no_match+1
                logging.info("lista lsh no match: " +str(no_match))

    while loop_i<220000:#300000:# and (match<totale or no_match<totale):
        #if loop_i%1000000==0:
            #logging.info("loop_i: "+str(loop_i))


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

        #serve per il conta occorrenza
        tableL_ELEM = concatenate_list_data(tableLlist[x]) #[ item for elem in tableLlist[x] for item in elem]
        tableR_ELEM = concatenate_list_data(tableRlist[y]) #[ item for elem in tableRlist[y] for item in elem]

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]>max_sim and match<3000:
                if (tableL_el,tableR_el,sim_vector) not in result_list:
                    #match
                    count_occurrence(dictL_match, tableL_ELEM)
                    count_occurrence(dictR_match, tableR_ELEM)

                    result_list.append((tableL_el,tableR_el,sim_vector))
                    match=match+1

                    #logging.info("lista random match: " +str(match)+" loop_i: "+str(loop_i))
                    loop_i=0
                else:
                    loop_i=loop_i+1
            elif sim_vector[0]<min_sim and no_match<(3000+tot_copy_match):
                if (tableL_el,tableR_el,sim_vector) not in result_list:
                    #NO_match
                    count_occurrence(dictL_NOmatch, tableL_ELEM)
                    count_occurrence(dictR_NOmatch, tableR_ELEM)

                    result_list.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    loop_i=0
                    #logging.info("lista random no match: " +str(no_match)+" loop_i: "+str(loop_i))
                else:
                    loop_i=loop_i+1

        elif copy_match<tot_copy_match:
            tableL_el2=copy_EDIT_match(tableL_el)
            sim_vector=sim_function(tableL_el,tableL_el2)
            if (tableL_el,tableL_el2,sim_vector) not in result_list and sim_vector[0]>max_sim:
                #match
                count_occurrence(dictL_match, tableL_ELEM)
                result_list.append((tableL_el,tableL_el2,sim_vector))
                copy_match_list.append((tableL_el,tableL_el2,sim_vector))
                copy_match=copy_match+1

                #logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                #logging.info(tableL_el,tableL_el2,sim_vector)
                loop_i=0
            #else:
            tableR_el2=copy_EDIT_match(tableR_el)
            sim_vector=sim_function(tableR_el,tableR_el2)
            if (tableR_el,tableR_el2,sim_vector) not in result_list and sim_vector[0]>max_sim:
                #match
                count_occurrence(dictR_match, tableR_ELEM)
                result_list.append((tableR_el,tableR_el2,sim_vector))
                copy_match_list.append((tableR_el,tableR_el2,sim_vector))
                copy_match=copy_match+1

                #logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                #logging.info(tableR_el,tableR_el2,sim_vector)
                loop_i=0

        elif no_match<(3000+tot_copy_match):
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]<min_sim and (tableL_el,tableR_el,sim_vector) not in result_list:
                #NO_match
                count_occurrence(dictL_NOmatch, tableL_ELEM)
                count_occurrence(dictR_NOmatch, tableR_ELEM)
                result_list.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                loop_i=0
                #logging.info("lista random no match wo cos_sim: " +str(no_match)+" loop_i: "+str(loop_i))
            else:
                loop_i=loop_i+1
        else:
            loop_i=loop_i+1



    plotting_dizionari(dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch)
    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list

def csvTable2datasetRANDOM_likeGold0(tableL,tableR,totale,min_sim,max_sim,indici,data_lsh,min_cos_sim, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    '''#soglia di cos similarità '''

    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')

    #skip header
    next(table1, None)
    next(table2, None)

    #convert to list for direct access
    tableLlist = list(table1)
    tableRlist = list(table2)
    no_match=0
    match=0
    result_list= []

    #totale=3000
    ##[1:] serve per togliere l id come attributo
    i=0
    #    for x in range(len(tableLlist)):
    #        for y in range(len(tableRlist)):
    for el in data_lsh:
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(el[0])
        stringa2=concatenate_list_data(el[1])
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            if el[2][0]>max_sim and el not in result_list:
                result_list.append(el)
                match=match+1
                logging.info("lista lsh match: " +str(match))
            if el[2][0]<min_sim and el not in result_list:
                result_list.append(el)
                no_match=no_match+1
                logging.info("lista lsh no match: " +str(no_match))

    while loop_i<150000:#300000:# and (match<totale or no_match<totale):
        #if loop_i%1000000==0:
            #logging.info("loop_i: "+str(loop_i))


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

        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
        if cos_sim>min_cos_sim:
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]>max_sim and match<2000:
                if (tableL_el,tableR_el,sim_vector) not in result_list:

                    result_list.append((tableL_el,tableR_el,sim_vector))
                    match=match+1

                    logging.info("lista random match: " +str(match)+" loop_i: "+str(loop_i))
                    loop_i=0
                else:
                    loop_i=loop_i+1
            elif sim_vector[0]<min_sim and no_match<2000:
                if (tableL_el,tableR_el,sim_vector) not in result_list:

                    result_list.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    loop_i=0
                    logging.info("lista random no match: " +str(no_match)+" loop_i: "+str(loop_i))
                else:
                    loop_i=loop_i+1
            elif copy_match<50:
                sim_vector=sim_function(tableL_el,tableL_el)
                if (tableL_el,tableL_el,sim_vector) not in result_list:
                    result_list.append((tableL_el,tableL_el,sim_vector))
                    copy_match=copy_match+1

                    logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    loop_i=0
                else:
                    sim_vector=sim_function(tableR_el,tableR_el)
                    if (tableR_el,tableR_el,sim_vector) not in result_list:
                        result_list.append((tableR_el,tableR_el,sim_vector))
                        copy_match=copy_match+1

                        logging.info("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                        loop_i=0

            else:
                loop_i=loop_i+1
    logging.info(result_list[:4])
    logging.info(result_list[-15:])
    return result_list

def parsing_anhai_nofilter(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1], cut=1):
    tableA = csv.reader(open(tableA, encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB, encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth, encoding="utf8"), delimiter=',')
    # skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)

    # convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    result_list = []

    if cut < 1:
        sl = int(max(len(trainFilelist)*cut, 5))
        trainFilelist = trainFilelist[:sl]

    for line_in_file in trainFilelist:
        # line_in_file type: id_1, id_2
        try:
            row1 = [item for item in tableAlist if item[0] == line_in_file[0]]
            row2 = [item for item in tableBlist if item[0] == line_in_file[1]]
            tableA_el = []
            tableB_el = []
            for i1, i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])

            # calcola la cos similarita della tupla i-esima
            sim_vector = sim_function(tableA_el, tableB_el)  # Modificato
            if int(line_in_file[2]) == 1:
                result_list.append((tableA_el, tableB_el, sim_vector, 1))
            else:
                result_list.append((tableA_el, tableB_el, sim_vector, 0))
        except:
            pass

    return result_list

