# -*- coding: utf-8 -*-
"""
@author: Aditya Vashista

This file is called in model.py and questions_reducer.py to add features to the input data 
of questions and also to standardize the data before model creation and predictions.

There are 4 functions to add total of 28 features on data set.
These 4 functions return tuple of 2: Modified data (pandas dataframe) & features added(list of column names)

There is a funtcion to standardise the data before giving it to models.
It return numpy matrix of standardised and clean input dataframe

There are 2 helper functions also
"""

#importing impotant libraries
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import gensim
import joblib

#genism model=first 8,00,000 enteries of GoogleNews-vectors-negative300.bin.gz
#Helper function to calculate Word Movers Distance using first 8,00,000 enteries of GoogleNews-vectors-negative300.bin.gz
def wmd(s1, s2,genismModel):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return genismModel.wmdistance(s1, s2)

#Helper function to change the strings into vectors to calculate various distances
def sent2vec(s,genismModel):
    words = str(s).lower()
    words = word_tokenize(words)
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(genismModel[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

#basic feature engineering
#These basic features include length-based features and string-based features
def data_transform(data): 
    # length based features
    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    # difference in lengths of two questions
    data['diff_len'] = data.len_q1 - data.len_q2
    
    # character length based features
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    
    # word length based features
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    
    # common words in the two questions
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1'])
            .lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    
    #For future reference, we will mark this set of features as feature set-1 or fs_1:
    fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1','len_char_q2', 'len_word_q1', 'len_word_q2','common_words']
    
    return data,fs_1


#Creating fuzzy features
#set of features are based on fuzzy string matching; Colums added are named in accordance of type of fuzzy feature stored
def fuzzy_features(data):
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])),axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']),str(x['question2'])), axis=1)
    
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), 
                                                                                             str(x['question2'])), axis=1)
    
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1'])
                                                                                               ,str(x['question2'])), axis=1)
    
    data['fuzz_token_set_ratio'] = data.apply(lambda x:fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    
    fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio','fuzz_partial_token_set_ratio', 
            'fuzz_partial_token_sort_ratio','fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
    
    return data,fs_2
    
#calculating word mover's distance and normalized word's mover distance using helper fnction: wmd()
def wmd_features(data,genismModel):
    data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'],genismModel), axis=1)
    genismModel.init_sims(replace=True)
    data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'],genismModel), axis=1)
    fs_3=['wmd','norm_wmd']
    return data,fs_3

#sequence of features with some measurement of the distribution of the two string vectors
#Vectors obtained by helper function: sec2vec()
    
def distance_features(data,genismModel):
    w2v_q1 = np.array([sent2vec(q, genismModel) for q in data.question1])
    w2v_q2 = np.array([sent2vec(q, genismModel) for q in data.question2])
    a=np.zeros(300)
    for i in range(len(w2v_q1)):
        if w2v_q1[i].size==1:
            w2v_q1[i]=a
    for i in range(len(w2v_q2)):
        if w2v_q2[i].size==1:
            w2v_q2[i]=a
    
    data['cosine_distance'] = [cosine(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['cityblock_distance'] = [cityblock(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['jaccard_distance'] = [jaccard(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['canberra_distance'] = [canberra(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['euclidean_distance'] = [euclidean(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['minkowski_distance'] = [minkowski(x,y,3) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['braycurtis_distance'] = [braycurtis(x,y) for (x,y) in zip(w2v_q1, w2v_q2)]
    data['skew_q1vec'] = [skew(x) for x in w2v_q1]
    data['skew_q2vec'] = [skew(x) for x in w2v_q2]
    data['kur_q1vec'] = [kurtosis(x) for x in w2v_q1]
    data['kur_q2vec'] = [kurtosis(x) for x in w2v_q2]
    fs_4 = ['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance', 
         'euclidean_distance', 'minkowski_distance','braycurtis_distance','skew_q1vec',
         'skew_q2vec','kur_q1vec','kur_q2vec']
    return data,fs_4
    

#Function to standardise final data frame
def data_standard(x):
    sc=joblib.load('models\sc.pkl')
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0).values
    x=sc.transform(x)
    return x