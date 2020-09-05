import os
import pandas as pd
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
stemmer = PorterStemmer()

def Prob(data,spam,ham,spam1,ham1):
    for word in data.keys():
        a_spam = (data[word][0]+1)/(spam+ len(data)*1)
        a_ham = (data[word][1]+1)/(ham + len(data)*1)
        data[word] = data[word] + [a_spam,a_ham]
    p_spam = spam1 / (spam1+ham1)
    p_ham = ham1 / (spam1+ham1)
    return data,p_spam,p_ham
