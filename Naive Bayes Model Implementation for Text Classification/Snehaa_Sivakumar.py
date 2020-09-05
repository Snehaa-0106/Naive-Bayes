import os
import pandas as pd
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import mytrain
import cond
import myaccuracy
import nltk
nltk.download('stopwords')
stemmer = PorterStemmer()

def predict(text,set1,spam,ham):
    file = open(text,errors="ignore")
    wordcount = Counter(file.read().split())
    spam_score = np.log(spam)
    ham_score = np.log(ham)
    stop_words = set(stopwords.words('english'))
    if myaccuracy.stop_words1 == 0:
        pass
    else:
        l = []
        for k in wordcount:
            if k in stop_words:
                l.append(k)
        for k in l:
            del wordcount[k]
    for item in wordcount.items():
        if item[0] in set1:
            spam_score += np.log(set1[item[0]][2])*item[1]
            ham_score += np.log(set1[item[0]][3])*item[1]
    if spam_score>=ham_score:
        return "spam"
    else:
        return "ham"


def main():
    c=0
    set1,spam,ham = mytrain.train()
    accuracy = myaccuracy.accuracy(set1,spam,ham)
    print("\nAccuracy of TRAINING data is \n",accuracy)
    set1,spam,ham = mytrain.test()
    accuracy1 = myaccuracy.accuracy1(set1,spam,ham)
    print("\nAccuracy of TEST data is \n",accuracy1)


if __name__ == "__main__":
    main()
