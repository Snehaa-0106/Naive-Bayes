import os
import pandas as pd
from collections import Counter
import numpy as np
import cond
from nltk.corpus import stopwords
from nltk.stem import *
stemmer = PorterStemmer()

def stem(data):
    new = {}
    for k,v in data.items():
        new[k] = stemmer.stem(k)
    for old,new in new.items():
        data[new] = data.pop(old)
    return data

# FOR TRAINING DATASET
def train():
    data = {}
    total_spam = 0
    total_ham = 0
    nof_spam = 0
    nof_ham = 0
    for filename in os.listdir('./ham_train'):
        file = open('./ham_train' + '//' + filename, errors='ignore')
        wordcount = Counter(file.read().split())
        for item in wordcount.items():
            if item[0] in data:
                data[item[0]][1] += item[1]
            else:
                data[item[0]] = [0,item[1]]
            total_ham += item[1]
        nof_ham += 1
    for filename in os.listdir('./spam_train'):
        file = open('./spam_train'+'//' +filename,errors='ignore')
        wordcount = Counter(file.read().split())
        for item in wordcount.items():
            if item[0] in data:
                 data[item[0]][0] += item[1]
            else:
                data[item[0]] = [item[1],0]
            total_spam += item[1]
        nof_spam += 1
    data = stem(data)
    data_new,ps,ph = cond.Prob(data,total_spam,total_ham,nof_spam,nof_ham)
    return data_new,ps,ph

# FOR TESTING DATASET
def test():
    data = {}
    total_spam = 0
    total_ham = 0
    nof_spam = 0
    nof_ham = 0
    for filename in os.listdir('./ham_test'):
        file = open('./ham_test' + '//' + filename, errors='ignore')
        wordcount = Counter(file.read().split())
        for item in wordcount.items():
            if item[0] in data:
                data[item[0]][1] += item[1]
            else:
                data[item[0]] = [0,item[1]]
            total_ham += item[1]
        nof_ham += 1
    for filename in os.listdir('./spam_test'):
        file = open('./spam_test'+'//' +filename,errors='ignore')
        wordcount = Counter(file.read().split())
        for item in wordcount.items():
            if item[0] in data:
                 data[item[0]][0] += item[1]
            else:
                data[item[0]] = [item[1],0]
            total_spam += item[1]
        nof_spam += 1
    #print(nof_spam)
    data = stem(data)
    data_new,ps,ph = cond.Prob(data,total_spam,total_ham,nof_spam,nof_ham)
    return data_new,ps,ph
