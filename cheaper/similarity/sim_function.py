import math
import re
from collections import Counter

import numpy
import scipy
import textdistance
import textdistance as txd
import torch
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram

WORD = re.compile(r'\w+')

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += ' '+str(element)
    return result

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

def norm(x,minim,maxi):
    z=(x-minim)/(maxi-minim)
    return z

def sim4attrScho(stringa1,stringa2):
    s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])
    
    t1_split=stringa1[1].split()
    t2_split=stringa2[1].split()
    s1= textdistance.jaccard.normalized_similarity(t1_split,t2_split)
    
    s2=get_cosine(text_to_vector(stringa1[2]),text_to_vector(stringa2[2]))
    s3=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    
    vect=[s0,s1,s2,s3]
    aver=round(sum(vect) / len(vect),2)
    return [aver]


def sim4attrFZ_norm2(stringa1,stringa2,minim,maxi):
    s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])
    
    t11_split=stringa1[1].split()
    t12_split=stringa2[1].split()
    s1= textdistance.jaccard.normalized_similarity(t11_split,t12_split)
    
    s2=get_cosine(text_to_vector(stringa1[2]),text_to_vector(stringa2[2]))
    
    t1_split4=stringa1[3].split()
    t2_split4=stringa2[3].split()
    s3= textdistance.jaccard.normalized_similarity(t1_split4,t2_split4)
    t41_split4=stringa1[4].split()
    t42_split4=stringa2[4].split()
    s4= textdistance.jaccard.normalized_similarity(t41_split4,t42_split4)
    
    s00=norm(s0,minim[0],maxi[0])
    s11=norm(s1,minim[1],maxi[1])
    s22=norm(s2,minim[2],maxi[2])
    s33=norm(s3,minim[3],maxi[3])
    s44=norm(s4,minim[4],maxi[4])
    
    vect=[s00,s11,s22,s33,s44]#,s5]
    
    #print(vect)
    rm_min=min(vect)
    vect.remove(rm_min)
    rm_max=max(vect)
    vect.remove(rm_max)
    aver=round(sum(vect) / len(vect),2)
    
    #print(aver)
    return [aver]


def sim4attrFZ_norm(stringa1,stringa2):
    s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])
    
    t1_split=stringa1[1].split()
    t2_split=stringa2[1].split()
    s1= textdistance.jaccard.normalized_similarity(t1_split,t2_split)
    
    s2=get_cosine(text_to_vector(stringa1[2]),text_to_vector(stringa2[2]))
    s3=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    t1_split4=stringa1[4].split()
    t2_split4=stringa2[4].split()
    s4= textdistance.jaccard.normalized_similarity(t1_split4,t2_split4)
   
    #s5=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    vect=[s0,s1,s2,s3,s4]#,s5]
    
    
    #print(vect)
    rm_min=min(vect)
    #vect.remove(rm_min)
    #print(vect)
    
    aver=round(sum(vect) / len(vect),2)
    
    #print(aver)
    #return [aver]
    return [aver],vect

def sim4attrFZ(stringa1,stringa2):
    s0=txd.jaro_winkler.normalized_similarity(stringa1[0],stringa2[0])
    
    t1_split=stringa1[1].split()
    t2_split=stringa2[1].split()
    s1= textdistance.jaccard.normalized_similarity(t1_split,t2_split)
    
    s2=get_cosine(text_to_vector(stringa1[2]),text_to_vector(stringa2[2]))
    s3=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    t1_split4=stringa1[4].split()
    t2_split4=stringa2[4].split()
    s4= textdistance.jaccard.normalized_similarity(t1_split4,t2_split4)
   
    #s5=textdistance.levenshtein.normalized_similarity(stringa1[3],stringa2[3])
    vect=[s0,s1,s2,s3,s4]#,s5]
    #print(vect)
    rm_min=min(vect)
    vect.remove(rm_min)
    #print(vect)
    
    aver=round(sum(vect) / len(vect),1)
    
    #print(aver)
    return [aver]

'''EDIT SIM'''

def sim_lev(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)

    lev = textdistance.levenshtein.normalized_similarity(t1_concat,t2_concat)
    vector=[lev]
    return vector

def jaro(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    jawi=txd.jaro_winkler.normalized_similarity(t1_concat,t2_concat)
    vector=[jawi]
    return vector

def sim_ngram(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    ngram = NGram()
    ngram1=1 - ngram.distance(t1_concat,t2_concat)
    vector=[ngram1]
    return vector   

def sim_lcs(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    metric_lcs = MetricLCS()
    lcs1=1 - metric_lcs.distance(t1_concat,t2_concat)
    vector=[lcs1]
    return vector

def sim_hamming(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    ham=txd.hamming.normalized_similarity(t1_concat,t2_concat)
    vector=[ham]
    return vector

def jacc_trigram(tupla1,tupla2):    
    sent1=concatenate_list_data(tupla1)
    sent2=concatenate_list_data(tupla2)
    
    ng1_chars = set(nltk.ngrams(sent1, n=3))
    ng2_chars = set(nltk.ngrams(sent2, n=3))
    
    jd_sent_1_2 =1- nltk.jaccard_distance(ng1_chars, ng2_chars)
    vector=[jd_sent_1_2]
    return vector

'''TOKEN SIM'''

import nltk

def sim_cos(tuple1,tuple2):
    stringa1=concatenate_list_data(tuple1)
    stringa2=concatenate_list_data(tuple2)
    cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
    vector=[cos_sim]
    return vector

def sim_jacc(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    t1_split=t1_concat.split()
    t2_split=t2_concat.split()
    jacc = textdistance.jaccard.normalized_similarity(t1_split,t2_split)
    vector=[jacc]
    return vector

def sim_sodi(tuple1,tuple2):
    t1_concat=concatenate_list_data(tuple1)
    t2_concat=concatenate_list_data(tuple2)
    t1_split=t1_concat.split()
    t2_split=t2_concat.split()
    sodi = textdistance.sorensen_dice.normalized_similarity(t1_split,t2_split)
    vector=[sodi]
    return vector

def jacc_trigramTOKEN(tupla1,tupla2):
    sent1=concatenate_list_data(tupla1)
    sent2=concatenate_list_data(tupla2)
    
    tokens1 = nltk.word_tokenize(sent1)
    tokens2 = nltk.word_tokenize(sent2)
    
    ng1_tokens = set(nltk.ngrams(tokens1, n=3))
    ng2_tokens = set(nltk.ngrams(tokens2, n=3))
    
    jd_sent_1_2 =1- nltk.jaccard_distance(ng1_tokens, ng2_tokens)
    vector=[jd_sent_1_2]
    return vector
    
def remove_symb(tupla):
    tupla1=[]
    for el in tupla:
        t=re.sub(r'[^\w]', ' ', str(el))
        tupla1.append(t)

    return tupla1

def min_cos(data):
    cosine=[]
    for el in data:
        stringa1=concatenate_list_data(el[0])
        stringa2=concatenate_list_data(el[1])
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        cosine.append(cos_sim)
        
    return min(cosine)


def sim_bert(stringa1, stringa2):
    _, _, e1 = extract_bert(' '.join(stringa1), tokenizer, model)
    _, _, e2 = extract_bert(' '.join(stringa2), tokenizer, model)

    a = torch.mean(e1, 0)
    a[torch.isnan(a)] = 0
    b = torch.mean(e2, 0)
    b[torch.isnan(b)] = 0

    v1 = numpy.nan_to_num(a.numpy())
    v2 = numpy.nan_to_num(b.numpy())
    distance = numpy.nan_to_num(scipy.spatial.distance.cosine(v1, v2))
    return [1 - distance]

def sim_sbert(stringa1, stringa2):
    e1 = embedder.encode([' '.join(stringa1)])
    e1 = numpy.nan_to_num(e1)
    e2 = embedder.encode([' '.join(stringa2)])
    e2 = numpy.nan_to_num(e2)
    return [1 - scipy.spatial.distance.cosine(e1, e2)]

def sim_sbert2(stringa1, stringa2):
    e1 = encoder.encode([' '.join(stringa1)])
    e1 = numpy.nan_to_num(e1)
    e2 = encoder.encode([' '.join(stringa2)])
    e2 = numpy.nan_to_num(e2)
    return [1 - scipy.spatial.distance.cosine(e1, e2)]

def extract_bert(text, tokenizer, model):
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
    states = []

    for ci in range(n_chunks):
        try:
            text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
            torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
            if text_ids[0, -1] != text_ids[0, -1]:
                torch.cat([text_ids, text_ids[0, -1].unsqueeze(0)])

            with torch.no_grad():
                state = model(text_ids_.unsqueeze(0))[0]
                state = state[:, 1:-1, :]
            states.append(state)
        except:
            pass
    state = torch.cat(states, axis=1)
    return text_ids, text_words, state[0]


'''PATH = "models" + os.sep + "sim_bert"

# bert
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.save_pretrained(PATH + os.sep + 'bert-encoder')
tokenizer.save_pretrained(PATH + os.sep + 'bert-encoder')

# s-bert
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# s-bert II
embedding = models.BERT('bert-base-uncased', max_seq_length=128, do_lower_case=True)
pooling_model = models.Pooling(embedding.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model2 = SentenceTransformer(modules=[embedding, pooling_model])
model2.save(PATH + os.sep + 'sbert-encoder2')
encoder = SentenceTransformer(PATH + os.sep + 'sbert-encoder2')'''