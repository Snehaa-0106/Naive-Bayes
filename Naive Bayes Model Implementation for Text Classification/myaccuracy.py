import os
import pandas as pd
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
import mytrain
import cond
import Snehaa_Sivakumar
stemmer = PorterStemmer()

# IF WE WANT TO FILTER OUT STOP WORDS, CHANGE STOP_WORDS = 1
stop_words1 = 0

#ACCURACY FOR THE TRAIN DATA
def accuracy(set1,spam1,ham1):
    p=0
    n=0
    p1=0
    n1=0
    path1 = r"./ham_train"
    for filename in os.listdir(path1):
        x = Snehaa_Sivakumar.predict(path1 + "//" + filename,set1,spam1,ham1)
        if x== "spam":
            n1 += 1
        if x == "ham":
            p += 1
    path2 = r"./spam_train"
    for filename in os.listdir(path2):
        x = Snehaa_Sivakumar.predict(path2 + "//" + filename,set1,spam1,ham1)
        if x == "spam":
            n += 1
        if x == "ham":
            p1 += 1
    if stop_words1 == 0:
        accuracy = (p+n)/(p+n+p1+n1)
        f = open("results.txt","a")
        f.write("\nBEFORE FILTERING THE STOP WORDS \n")
        f.write("The accuracy of the TRAIN data is %.4f\n" %accuracy)
        f.close()
    else:
        accuracy = (p+n)/(p+n+p1+n1)
        f = open("results.txt","a")
        f.write("\nAFTER FILTERING THE STOP WORDS \n")
        f.write("The accuracy of the TRAIN data after filtering out the stop words is %.4f\n" %accuracy)
        f.close()
    return accuracy

# ACCURACY FOR THE TEST DATA

def accuracy1(set2,spam2,ham2):
    p=0
    n=0
    p1=0
    n1=0
    path1 = r'./ham_test'
    for filename in os.listdir(path1):
        x = Snehaa_Sivakumar.predict(path1 + "//" + filename,set2,spam2,ham2)
        if x== "spam":
            n1 += 1
        if x == "ham":
            p += 1
    path2 = r"./spam_test"
    for filename in os.listdir(path2):
        x = Snehaa_Sivakumar.predict(path2 + "//" + filename,set2,spam2,ham2)
        if x == "spam":
            n += 1
        if x == "ham":
            p1 += 1
    if stop_words1 == 0:
        accuracy1 = (p+n)/(p+n+p1+n1)
        f = open("results.txt","a")
        f.write("The accuracy of the TEST data is %.4f\n" %accuracy1)
        f.close()
    else:
        accuracy1 = (p+n)/(p+n+p1+n1)
        f = open("results.txt","a")
        f.write("The accuracy of the TEST data after filtering out the stop words is %.4f\n\n" %accuracy1)
        f.close()
    return accuracy1
