import os
import pandas as pd
from collections import Counter
import numpy as np
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

def train():
    data = {}
    total_spam = 0
    total_ham = 0
    nof_spam = 0
    nof_ham = 0
    for filename in os.listdir("./ham_train"):
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
    return data
