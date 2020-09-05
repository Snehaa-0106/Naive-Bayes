import os
import pandas as pd
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
import mytrain
import remove_stopword
stemmer = PorterStemmer()
# CHANGE STOPWORD = 1 , IF YOU WANT TO REMOVE STOP WORDS!!!
stopword = 0

def find(instance,weights):
    weight_sum = weights['weight_zero']
    file = open('./Testing'+'//' +instance,errors='ignore')
    wordcount = Counter(file.read().split())
    if (stopword):
         wordcount = remove_stopword.rem_stop(wordcount)
    for f in wordcount.keys():
        if f not in weights:
            weights[f] = 0.0
        weight_sum += weights[f]*wordcount[f]
    if weight_sum>0:
        return 1
    else:
        return 0

def learnweights(learning_const,n):
    ni = n
    weights = {'weight_zero':1}
    d1 = mytrain.train()
    for i in d1.keys():
            weights[i] = 0.0
    for i in range(0,ni):
        for filename in os.listdir('./Training'):
            k = 'spam'
            file = open('./Training'+'//' +filename,errors='ignore')
            wordcount = Counter(file.read().split())
            weight_sum = weights['weight_zero']
            if (stopword):
                 wordcount = remove_stopword.rem_stop(wordcount)
            for f in wordcount.keys():
                if f not in weights:
                    weights[f] = 0.0
                weight_sum += weights[f]*wordcount[f]
            perceptron_output = 0.0
            if weight_sum>0:
                perceptron_output = 1.0
            target_value = 0.0
            if remove_stopword.isClass(filename) == 'spam':
                target_value = 1.0
            for w in wordcount.keys():
                weights[w] += float(learning_const)*float((target_value - perceptron_output))*float(wordcount[w])
    corr_guess = 0
    count = 0.0
    for filename in os.listdir('./Testing'):
        guess = find(filename,weights)
        if guess == 1 :
            if remove_stopword.isClass(filename) == 'spam':
                corr_guess += 1
        if guess == 0:
            if remove_stopword.isClass(filename) == 'ham':
                corr_guess += 1
        count += 1
    accuracy = float(corr_guess)/float(count)
    print(accuracy)

def main():
    NoI = 5
    for i in [float(j)/100 for j in range(5,105,5)]:
        learnweights(i,NoI)
        NoI += 5
if __name__ == "__main__":
    main()
