import os
import pandas as pd
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
import mytrain
stemmer = PorterStemmer()

def rem_stop(wordcount):
    l = []
    stop_words = set(stopwords.words('english'))
    for k in wordcount:
        if k in stop_words:
            l.append(k)
    for k in l:
        del wordcount[k]
    return wordcount

def isClass(filename):
    output = ""
    if 'spam' in filename:
        output = 'spam'
    if 'ham' in filename:
        output = 'ham'
    return output
