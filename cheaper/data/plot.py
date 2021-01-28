import matplotlib.pyplot as plt
plt.switch_backend('agg')
from itertools import islice 

# Ti restituisce i primi n valori di datatasetOriginal
# con n = len(datasetOriginal) * percentuale
def splitting_dataSet(percentuale, dataSetOriginal):
    lunghezza=int(len(dataSetOriginal)*percentuale)
    lunghezza = max(5, lunghezza)
    "Return first n items of the iterable as a list"
    output=list(islice(dataSetOriginal, lunghezza))
     
    #print("Split length list: ", percentuale) 
    #print("List after splitting", output)
    return output

def plotting(result_list,index):
    sim_list=[]
    label_list=[]
    t=[]
    
    sim_listArr=[]
    
    sorted_by_secondEthird = sorted(result_list, key=lambda tup: (tup[3], tup[2][index]))
    for i in range(len(sorted_by_secondEthird)):
        
        sim_list.append(sorted_by_secondEthird[i][2][index])
        label_list.append(sorted_by_secondEthird[i][3])
        t.append(i)
        
    index_min=label_list.index(1)
    min_sim_match=sim_list[index_min]
    max_sim_noMatch=sim_list[index_min-1]
    average=(sum(sim_list) / len(sim_list))
    print(average)
    wrong_match=0
    wrong_NOmatch=0
    for i in range(len(sim_list)):
        
        if sim_list[i]>=average:
            if label_list[i]!=1:
                wrong_match=wrong_match+1
            sim_listArr.append(1)
        else:
            sim_listArr.append(0)
            if label_list[i]!=0:
                wrong_NOmatch=wrong_NOmatch+1
    
    plt.plot(t, label_list, '-b',t, sim_list, '-r')
    plt.ylabel('plot_sim'+str(index))
    plt.show(block=False)
    '''
    plt.plot(t, label_list, '-b',t, sim_listArr, '-r' )
    plt.ylabel('plot_simArr'+str(index))
    plt.show()
    
    print("wrong_match. "+str(wrong_match)) 
    print("wrong_NOmatch. "+str(wrong_NOmatch))    
    print(min_sim_match)'''
    return min_sim_match,max_sim_noMatch
    #return sim_list, label_list, t, sim_listArr

    

    
def plot_graph(result_list,cut):
   
    for j in range(len(result_list[0][2])):
        print(j)
#        result_listANHAI1=ratio_dupl_noDup4Anhai(result_list,j)
#        shuffle(result_listANHAI1)
        dataset5Percent=splitting_dataSet(cut, result_list)
        g=0
        k=0
        for i in range(len(dataset5Percent)):
            if dataset5Percent[i][3]==1:
                g=g+1
            else:
                k=k+1
        
        print("match number: "+str(g)+ " no match number: " + str(k))
        min_sim_match,max_sim_noMatch=plotting(dataset5Percent,j)
    return min_sim_match,max_sim_noMatch
def plot_pretrain(data):
    random_tuples1 = data[:1000]
    random_tuples2 = data[-1000:]


    random_tuples1 +=random_tuples2
    result_list=[]
    result_list = sorted(data, key=lambda tup: (tup[2][0]))
    
    
    sim_list=[]
    t=[]
    for i in range(len(result_list)):
        
        sim_list.append(result_list[i][2][0])
        t.append(i)
    plt.plot(t, sim_list, '-r')
    plt.ylabel('plot_pretraining data')
    plt.show(block=False)
    
def plot_dataPT(data):
    result_list = sorted(data, key=lambda tup: (tup[2][0]))
    sim_list = []
    t = []
    for i in range(len(result_list)):
        sim_list.append(result_list[i][2][0])
        t.append(i)
    plt.plot(t, sim_list, '-r')
    gradino = []
    for g in range(len(t)):
        if g >= len(t) / 2:
            gradino.append(1)
        else:
            gradino.append(0)
    plt.plot(t, gradino)
    plt.ylabel('pretraining dataset')
    plt.show(block=False)
    return t, sim_list
    
def plotting_dizionari(dictL_match, dictR_match, dictL_NOmatch, dictR_NOmatch) :
    listL_match=list(dictL_match.values())
    print("listL_match[0]")
    print(listL_match[0])
    plotting_occorrenze(listL_match, "dictL_match")
    
    listR_match=list(dictR_match.values())
    plotting_occorrenze(listR_match, "dictR_match")
    
    listL_NOmatch=list(dictL_NOmatch.values())
    plotting_occorrenze(listL_NOmatch, "dictL_NOmatch")
    
    listR_NOmatch=list(dictR_NOmatch.values())
    plotting_occorrenze(listR_NOmatch, "dictR_NOmatch")
    
    
    
def plotting_occorrenze(data, stringa_plotting):
    data.sort() 
    t=list(range(0, len(data)))
    
    plt.plot(t, data, '-r')
    plt.ylabel(stringa_plotting)
    plt.show(block=False)
    
