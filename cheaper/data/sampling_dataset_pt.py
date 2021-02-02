import csv
from datasketch import MinHash, MinHashLSH

import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import re, math
from collections import Counter

from cheaper.data.edit_dna import Sequence
from cheaper.data.plot import plot_pretrain, plotting_dizionari, plot_dataPT

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
    #print(df)
    cos_sim=cosine_similarity(df,df)
    #print(cos_sim[0][-1])
    return cos_sim[0][-1]


def create_data(tableL, tableR, indiciL,indiciR):
   table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
   table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',') 
   next(table1, None)
   next(table2, None)
    
   #convert to list for direct access
   tableLlist = list(table1)
   tableRlist = list(table2)
   
    
   result_list= []
   result_list1,dataL=sampling_table(tableLlist,indiciL)
   result_list.extend(result_list1)
   result_list2,dataR=sampling_table(tableRlist,indiciR)
   result_list.extend(result_list2)
   return result_list,dataL,dataR,tableLlist,tableRlist

def sampling_table(table_list,indici):
   result_list1=[]
   data=[]
   for j in range(len(table_list)):
       table_el=[]
       for i1 in indici:
           table_el.append(table_list[j][i1])
       data.append(table_el)
       stringa_el=concatenate_list_data(table_el)
       lista_di_stringhe=stringa_el.split()
       result_list1.append(lista_di_stringhe)
   return result_list1,data  
   


def minHash_LSH(data):
    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    # Create LSH index
    lsh = MinHashLSH(threshold=0.65, num_perm=256)
    
    # Create MinHash objects
    minhashes = {}
    for c, i in enumerate(data):
      #c è l'indice, i è la tupla
      #print(i)
      minhash = MinHash(num_perm=256)
      for el in i:
          minhash.update(el.encode('utf8'))
#      for d in ngrams(i, 3):
#        minhash.update("".join(d).encode('utf-8'))
      lsh.insert(c, minhash)
      minhashes[c] = minhash
      #print(str(c)+" "+str(minhashes[c]))
      
    res_match=[]
    for i in range(len(minhashes.keys())):
      result = lsh.query(minhashes[i])
      
      if result not in res_match and len(result)==2:
          res_match.append(result)
          #print("Candidates with Jaccard similarity > 0.6 for input", i, ":", result)
    #print(res)
#    for i in range(len(res_match)):
#        print(data[res_match[i][0]])
#        print(data[res_match[i][1]])
    return res_match

def create_dataset_pt(res, dataL,dataR,tableLlist,tableRlist,min_sim,max_sim,dictL_match,dictR_match,dictL_NOmatch,dictR_NOmatch,sim_function):
#    indL=len(dataL)-1
#    indR=len(dataR)-1
    dataPT=[]
    i=0
    #print("lunghezza lsh"+ str(len(res)))
    for el in res:
       el1,table1,index1=find_el(el[0],dataL,dataR) 
       el2,table2,index2=find_el(el[1],dataL,dataR)
       #print("creo dataset da lsh"+str(i))
       i=i+1
       if table1!=table2:
           #print("controllo table1!=table2")
       
           if table1=="L":
               tableL_ELEM = concatenate_list_data(tableLlist[index1])
               tableR_ELEM = concatenate_list_data(tableRlist[index2])
           else:
               tableR_ELEM = concatenate_list_data(tableRlist[index1])
               tableL_ELEM = concatenate_list_data(tableLlist[index2])
       
           sim_vector=sim_function(el1,el2)
           if sim_vector[0]>max_sim:
               #match
               if count_occurrence(dictL_match, tableL_ELEM) and count_occurrence(dictR_match, tableR_ELEM):
                   dataPT.append((el1,el2,sim_vector))
           if sim_vector[0]<min_sim:
               #NO_match
               if count_occurrence(dictL_NOmatch, tableL_ELEM) and count_occurrence(dictR_NOmatch, tableR_ELEM):
                   dataPT.append((el1,el2,sim_vector))
                   
           #dataPT.append((el1,el2,sim_vector))
    #print(dataPT)
    return dataPT    

def find_el(index,dataL,dataR):
    #serve per il conta occorrenza
    #tableL_ELEM = concatenate_list_data(tableLlist[x]) #[ item for elem in tableLlist[x] for item in elem]
    #tableR_ELEM = concatenate_list_data(tableRlist[y]) #[ item for elem in tableRlist[y] for item in elem]
    if index>=len(dataL):
        indR=index-len(dataL)
        data_el=dataR[indR]
        table="R"
        return data_el,table,indR
    else:
        data_el=dataL[index]
        table="L"
        return data_el,table,index

def split_indici(indici):
    indiciL=[]
    indiciR=[]
    for i in range(len(indici)):
        indiciL.append(indici[i][0])
        indiciR.append(indici[i][1])
    return indiciL,indiciR

def minHash_lsh(tableL, tableR, indici,min_sim,max_sim,dictL_match,dictR_match,dictL_NOmatch,dictR_NOmatch,sim_function):
    indiciL,indiciR=split_indici(indici)
    data4hash,dataL,dataR,tableLlist,tableRlist=create_data(tableL, tableR, indiciL,indiciR)
    res=minHash_LSH(data4hash)
    dataset_pt=create_dataset_pt(res, dataL,dataR,tableLlist,tableRlist,min_sim,max_sim,dictL_match,dictR_match,dictL_NOmatch,dictR_NOmatch,sim_function)
    print("LSH blocking done")
    plot_dataPT(dataset_pt)

    return dataset_pt

def copy_EDIT_match(tupla):
    copy_tup=[]
    
    for i in range(len(tupla)):
        change_attr=random.randint(0,2)
        attr=Sequence(tupla[i])    
        if len(tupla[i])>1 and change_attr==1:
            #print(attr)
            d = 1  # max edit distance
            n = 3  # number of strings in result
            mutates=attr.mutate(d, n)
            #print(mutates[1])
            
            copy_tup.append(str(mutates[1]))
        else:
            copy_tup.append(tupla[i])
        
    #print(copy_tup)
    return copy_tup
    

    
def create_flat_list(lista):
    flatList = []
    for el in lista:
        tableELEM = concatenate_list_data(el)
        flatList.append(tableELEM)
    return flatList
 
def dict_tuple(csv_list):
    flatList=create_flat_list(csv_list)
    #print("elemento 0 della flat list")
    #print(flatList[0])
    dictTuple=dict((el,0) for el in flatList)
    return dictTuple        


def count_occurrence(dizion, tupla, limit='8'):
        if dizion[tupla] <int(limit):
            
            dizion[tupla] += 1
            return True
        else:
            return False


def csvTable2datasetRANDOM_countOcc(tableL,tableR,totale,min_sim,max_sim,indici,min_cos_sim,tot_copy_match, max_occ, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
        
    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')   
  
    #skip header
    next(table1, None)
    next(table2, None)
   
    #convert to list for direct access
    tableLlist = list(table1)
    print(len(tableLlist))
    tableRlist = list(table2)
    print(len(tableRlist))
    #print("sono il csv riga 0")
    #print(tableLlist[0])
    #create dict for count the occorrence
    
    
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
    #dictR_NOmatch=dict((el,0) for el in tableRlist)
    
    no_match=0
    match=0
    result_list_noMatch= []
    result_list_match=[]
    copy_match_list=[]
    
    data_lsh=minHash_lsh(tableL, tableR, indici,min_sim,max_sim,dictL_match,dictR_match,dictL_NOmatch,dictR_NOmatch,sim_function)
    print("ritornato insieme proveniente da minHash lsh")
    for el in data_lsh:
        if el[2][0]>max_sim and el not in result_list_match:
            result_list_match.append(el)
                
#                match=match+1
#                print("lista lsh match: " +str(match))
#            if el[2][0]<min_sim and el not in result_list:
#                result_list.append(el)
#                no_match=no_match+1
#                print("lista lsh no match: " +str(no_match))

    count_i = 0
    while loop_i<120000 and (match<totale or no_match<totale):
        
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
                if (tableL_el,tableR_el,sim_vector) not in result_list_match:
                    #match
                    if count_occurrence(dictL_match, tableL_ELEM) and count_occurrence(dictR_match, tableR_ELEM):
                        
                        #count_occurrence(dictL_match, tableL_ELEM)
                        #count_occurrence(dictR_match, tableR_ELEM)
                        
                        result_list_match.append((tableL_el,tableR_el,sim_vector))
                        match=match+1
    
                        #print("lista random match: " +str(match)+" loop_i: "+str(loop_i))
                        loop_i=0
                    else:
                        loop_i=loop_i+1
            elif sim_vector[0]<min_sim and no_match<(3000+tot_copy_match):
                if (tableL_el,tableR_el,sim_vector) not in result_list_noMatch:
                    #NO_match
                    if count_occurrence(dictL_NOmatch, tableL_ELEM, limit=max_occ) and count_occurrence(dictR_NOmatch, tableR_ELEM, limit=max_occ):
                        #NO_match
                        #count_occurrence(dictL_NOmatch, tableL_ELEM)
                        #count_occurrence(dictR_NOmatch, tableR_ELEM)
    
                        result_list_noMatch.append((tableL_el,tableR_el,sim_vector))
                        no_match=no_match+1
                        loop_i=0
                        #print("lista random no match: " +str(no_match)+" loop_i: "+str(loop_i))
                    else:
                        loop_i=loop_i+1
                    
        elif copy_match<tot_copy_match:
            tableL_el2=copy_EDIT_match(tableL_el)
            sim_vector=sim_function(tableL_el,tableL_el2)
            if (tableL_el,tableL_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                if count_occurrence(dictL_match, tableL_ELEM):
                    
                    #count_occurrence(dictL_match, tableL_ELEM)
                    result_list_match.append((tableL_el,tableL_el2,sim_vector))
                    copy_match_list.append((tableL_el,tableL_el2,sim_vector))
                    copy_match=copy_match+1
    
                    #print("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    #print(tableL_el,tableL_el2,sim_vector)
                    loop_i=0
            
            tableR_el2=copy_EDIT_match(tableR_el)
            sim_vector=sim_function(tableR_el,tableR_el2)
            if (tableR_el,tableR_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                if count_occurrence(dictR_match, tableR_ELEM, limit=max_occ):
                    #count_occurrence(dictR_match, tableR_ELEM)
                    result_list_match.append((tableR_el,tableR_el2,sim_vector))
                    copy_match_list.append((tableR_el,tableR_el2,sim_vector))
                    copy_match=copy_match+1
    
                    #print("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    #print(tableR_el,tableR_el2,sim_vector)
                    loop_i=0

        elif no_match<(3000+tot_copy_match):
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]<min_sim and (tableL_el,tableR_el,sim_vector) not in result_list_noMatch:
                #NO_match
                if count_occurrence(dictL_NOmatch, tableL_ELEM, limit=max_occ) and count_occurrence(dictR_NOmatch, tableR_ELEM, limit=max_occ):
                    
                    #count_occurrence(dictL_NOmatch, tableL_ELEM)
                    #count_occurrence(dictR_NOmatch, tableR_ELEM)
                    result_list_noMatch.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    loop_i=0
                #print("lista random no match wo cos_sim: " +str(no_match)+" loop_i: "+str(loop_i))
            else:
                loop_i=loop_i+1
        else:
            loop_i=loop_i+1
        count_i += 1
            
    
    print("dizionari")   
    plotting_dizionari(dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch)       
    print("create candidates set")
    return result_list_noMatch,result_list_match


#metodo senza il limite delle occorrenze
def csvTable2datasetRANDOM_NOOcc(tableL,tableR,totale,min_sim,max_sim,indici,min_cos_sim,tot_copy_match, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
        
    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')   
  
    #skip header
    next(table1, None)
    next(table2, None)
   
    #convert to list for direct access
    tableLlist = list(table1)
    print(len(tableLlist))
    tableRlist = list(table2)
    print(len(tableRlist))
    #print("sono il csv riga 0")
    #print(tableLlist[0])
    #create dict for count the occorrence
    
    
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
    #dictR_NOmatch=dict((el,0) for el in tableRlist)
    
    no_match=0
    match=0
    result_list_noMatch= []
    copy_match_list=[]
    
    result_list_match=minHash_lsh(tableL, tableR, indici,min_sim,max_sim,dictL_match,dictR_match,dictL_NOmatch,dictR_NOmatch,sim_function)
#    for el in data_lsh:
#        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
#        stringa1=concatenate_list_data(el[0])
#        stringa2=concatenate_list_data(el[1])
#        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
#        
#        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min_cos_sim definito sopra
#        if cos_sim>min_cos_sim:    
#            if el[2][0]>max_sim and el not in result_list:
#                result_list.append(el)
#                
#                match=match+1
#                print("lista lsh match: " +str(match))
#            if el[2][0]<min_sim and el not in result_list:
#                result_list.append(el)
#                no_match=no_match+1
#                print("lista lsh no match: " +str(no_match))
                
    while loop_i<120000:#300000:# and (match<totale or no_match<totale):
        #if loop_i%1000000==0:
            #print("loop_i: "+str(loop_i))
        
        
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
                if (tableL_el,tableR_el,sim_vector) not in result_list_match:
                    #match
                    result_list_match.append((tableL_el,tableR_el,sim_vector))
                    match=match+1
    
                    #print("lista random match: " +str(match)+" loop_i: "+str(loop_i))
                    loop_i=0
                else:
                    loop_i=loop_i+1
            elif sim_vector[0]<min_sim and no_match<(3000+tot_copy_match):
                if (tableL_el,tableR_el,sim_vector) not in result_list_noMatch:
                    #NO_match
                    result_list_noMatch.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    loop_i=0
                    
                else:
                    loop_i=loop_i+1
                    
        elif copy_match<tot_copy_match:
            tableL_el2=copy_EDIT_match(tableL_el)
            sim_vector=sim_function(tableL_el,tableL_el2)
            if (tableL_el,tableL_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                result_list_match.append((tableL_el,tableL_el2,sim_vector))
                copy_match_list.append((tableL_el,tableL_el2,sim_vector))
                copy_match=copy_match+1
    

                loop_i=0
            
            tableR_el2=copy_EDIT_match(tableR_el)
            sim_vector=sim_function(tableR_el,tableR_el2)
            if (tableR_el,tableR_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                result_list_match.append((tableR_el,tableR_el2,sim_vector))
                copy_match_list.append((tableR_el,tableR_el2,sim_vector))
                copy_match=copy_match+1
    
                
                loop_i=0

        elif no_match<(3000+tot_copy_match):
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]<min_sim and (tableL_el,tableR_el,sim_vector) not in result_list_noMatch:
                #NO_match
                result_list_noMatch.append((tableL_el,tableR_el,sim_vector))
                no_match=no_match+1
                loop_i=0
                
            else:
                loop_i=loop_i+1
        else:
            loop_i=loop_i+1
            
        
    print("create candidates set")
    return result_list_noMatch,result_list_match

def csvTable2datasetRANDOM_sciocco(tableL,tableR,totale,min_sim,max_sim,indici,min_cos_sim,tot_copy_match, sim_function=lambda x, y: [1, 1] ):
    #senza doppioni nella lista
    
    tot_copy_match=int(totale/2)
        
    loop_i=0
    copy_match=0
    table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',')   
  
    #skip header
    next(table1, None)
    next(table2, None)
   
    #convert to list for direct access
    tableLlist = list(table1)
    print(len(tableLlist))
    tableRlist = list(table2)
    print(len(tableRlist))
    #print("sono il csv riga 0")
    #print(tableLlist[0])
    #create dict for count the occorrence
    
    
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
    #dictR_NOmatch=dict((el,0) for el in tableRlist)
    
    no_match=0
    match=0
    result_list_noMatch= []
    result_list_match=[]
    copy_match_list=[]
    
                
    while loop_i<120000:#300000:# and (match<totale or no_match<totale):
        #if loop_i%1000000==0:
            #print("loop_i: "+str(loop_i))
        
        
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
        

                
        if copy_match<tot_copy_match:
            tableL_el2=copy_EDIT_match(tableL_el)
            sim_vector=sim_function(tableL_el,tableL_el2)
            if (tableL_el,tableL_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                if count_occurrence(dictL_match, tableL_ELEM):
                    
                    #count_occurrence(dictL_match, tableL_ELEM)
                    result_list_match.append((tableL_el,tableL_el2,sim_vector))
                    copy_match_list.append((tableL_el,tableL_el2,sim_vector))
                    copy_match=copy_match+1
    
                    #print("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    #print(tableL_el,tableL_el2,sim_vector)
                    loop_i=0
            
            tableR_el2=copy_EDIT_match(tableR_el)
            sim_vector=sim_function(tableR_el,tableR_el2)
            if (tableR_el,tableR_el2,sim_vector) not in result_list_match and sim_vector[0]>max_sim:
                #match
                if count_occurrence(dictR_match, tableR_ELEM):
                    #count_occurrence(dictR_match, tableR_ELEM)
                    result_list_match.append((tableR_el,tableR_el2,sim_vector))
                    copy_match_list.append((tableR_el,tableR_el2,sim_vector))
                    copy_match=copy_match+1
    
                    #print("lista copy_match match: " +str(copy_match)+" loop_i: "+str(loop_i))
                    #print(tableR_el,tableR_el2,sim_vector)
                    loop_i=0

        elif no_match<(tot_copy_match):
            sim_vector=sim_function(tableL_el,tableR_el)
            if sim_vector[0]<min_sim and (tableL_el,tableR_el,sim_vector) not in result_list_noMatch:
                #NO_match
                if count_occurrence(dictL_NOmatch, tableL_ELEM) and count_occurrence(dictR_NOmatch, tableR_ELEM):
                    
                    #count_occurrence(dictL_NOmatch, tableL_ELEM)
                    #count_occurrence(dictR_NOmatch, tableR_ELEM)
                    result_list_noMatch.append((tableL_el,tableR_el,sim_vector))
                    no_match=no_match+1
                    loop_i=0
                #print("lista random no match wo cos_sim: " +str(no_match)+" loop_i: "+str(loop_i))
            else:
                loop_i=loop_i+1
        else:
            loop_i=loop_i+1
            
    
    print("dizionari")   
    plotting_dizionari(dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch)       
    print("create candidates set")
    return result_list_noMatch,result_list_match







# TEST AREA #
if __name__ == "__main__":
    from cheaper.similarity.sim_function import sim4attrFZ#,sim4attrFZ_norm,sim4attrFZ_norm2

    #tableL='beer_exp_data/exp_data/tableA.csv'
    #tableR='beer_exp_data/exp_data/tableB.csv'
    TABLE1_FILE='fodo_zaga/fodors.csv'
    TABLE2_FILE='fodo_zaga/zagats.csv'
    ATT_INDEXES=[(1, 1), (2, 2), (3, 3),(4,4),(5,5),(6,6)]
#    indiciL=[1,2,3,5]
#    indiciR=[1,2,3,5]
    min_sim=0.4
    max_sim=0.85
    min_cos_sim=0.2
    tot_copy=1000
    
    simf = lambda a, b: sim4attrFZ(a, b)
    random_tuples0 =csvTable2datasetRANDOM_countOcc(TABLE1_FILE,TABLE2_FILE,7000,min_sim,max_sim,ATT_INDEXES,min_cos_sim, tot_copy, simf )
    random.shuffle(random_tuples0)
    random_tuples0sort = sorted(random_tuples0, key=lambda tup: (tup[2][0]))
    print("---------------- RANDOM TUPLES -------------------------")
    plot_pretrain(random_tuples0sort)
    
    
    
    #dataset_pt=minHash_lsh(tableL, tableR, indici,simf)
    #tableL='walmart_amazon/walmart.csv'
    #tableR='walmart_amazon/amazonw.csv'
    #indiciL=[5,4,3,14,6]
    #indiciR=[9,5,3,4,11]
    
    #data4hash,dataL,dataR=create_data(tableL, tableR, indiciL,indiciR)
    #res=minHash_LSH(data4hash)
    #dataset_pt=create_dataset_pt(res, dataL,dataR,sim4attrFZ)
    #print(dataset_pt[10:])
    
    #data = ['minhash is a probabilistic data structure for estimating the similarity between datasets',
    #  'finhash dis fa frobabilistic fata ftructure for festimating the fimilarity fetween fatasets',
    #  'weights controls the relative importance between minizing false positive',
    #  'wfights cfntrols the rflative ifportance befween minizing fflse posftive',
    #  "arnie morton\\'s of chicago 435 s. la cienega blv. los angeles 310/246-1501 american 0", 
    #  "arnie morton\\'s of chicago 435 s. la cienega blvd. los angeles 310-246-1501 steakhouses 0"]
    #plot_dataPT(dataset_pt)