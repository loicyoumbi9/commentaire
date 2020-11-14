from os import listdir
import os
#import re
import string
from os.path import isfile, join, isdir
from pathlib import Path

#import json
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
#from nltk.tokenize import word_tokenize
#import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2)
import itertools
import random
import sys


#from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense

def reste_ligne(nb_ligne_fichier, nb_vecteur, nb_ligne_para): #pour obtenir le reste de ligne lorsqu'on divise le fichier en paragraphe
    ligne_restant = nb_ligne_fichier - (nb_vecteur*nb_ligne_para)
    return ligne_restant


def nombre_de_ligne(fichier_a_compter, chemin):
    nombre_de_ligne = 0
    lines = open(chemin+fichier_a_compter, 'r').readlines()
    for line in lines:
        nombre_de_ligne += 1
    return nombre_de_ligne

def nom_fichier(chemin):
    fichier = [f for f in listdir(chemin) if isfile(join(chemin, f))]
    return fichier


def readline(n, lines): # pour lire une ligne a travers le numero de ligne
    for lineno, line in enumerate(lines):
        return line
        
        
def bloc_fichier_en_tableau(element, chemin): # mettre les paragraĥes dans un tableau
    tab = []
    content = []
    n = 0
    i = 0
    k = 0
    j = 0
    tableau = []
    line = []
    tab_reste = []
    lines = ""
    
    #for element in nom_fichier('dossier/'):
    #for element in nom_fichier(path):
    nb_ligne_fichier = nombre_de_ligne( element, chemin)
    nb_vecteur = 300
    nb_ligne_para = (nb_ligne_fichier // nb_vecteur) 
    m = (nb_ligne_fichier // nb_vecteur) 
    ligne_restant = reste_ligne(nb_ligne_fichier, nb_vecteur, nb_ligne_para)
        
    #lines = open('dossier/'+element, 'r')
    lines = open(chemin+element, 'r')
        
    while (i <= nb_ligne_fichier and nb_ligne_para  <= nb_ligne_fichier):
        for k in range(i, nb_ligne_para ) :
            tab +=readline(k,lines).split()
        content.append(tab)	           	
        i+=m
        nb_ligne_para += m
        tab = []
        
    if ligne_restant != 0:
        for j in range(i,nb_ligne_fichier):
            tab_reste +=readline(j,lines).split()
    #print(tab_reste )
        
    content.append(tab_reste) 
    tableau += content
    content = []
    tab = []
    i = 0
    nb_ligne_para = 0
    #print(len(tableau))
    #print("\n")
    
    return tableau
    
   



def benigne():
    tableau_benigne_final = []
    path = "begnine_traite/"
    i = 0
    b = []
    tableau_benigne = []
    #benigne_label =np.array([i for i in range(0,10)])
    
    model = Doc2Vec.load("epochs_50/para2vec.kv")
    for element in nom_fichier(path):
        test_doc = bloc_fichier_en_tableau(element, path)
        for e in test_doc:
            #a = model.docvecs.most_similar(positive=[model.infer_vector(e)],topn=5)
            b = model.infer_vector(e)
            for i in range(0, len(b)-1):
                tableau_benigne.append(b[i])
        
        tableau_benigne_final.append(np.array(tableau_benigne))
        b=[]
        tableau_benigne = []
    print(tableau_benigne_final)    
    return np.array(tableau_benigne_final)


 
 


def malware():
    tableau_malware_final = [] 
    path_malware =  "malware_traite/"  
    i = 0
    b = []
    tableau_benigne = []
    #benigne_label = np.array([i for i in range(0,10)])
    
    model = Doc2Vec.load("epochs_50/para2vec.kv")
    for element in nom_fichier(path_malware):
        test_doc = bloc_fichier_en_tableau(element, "malware_traite/" )
        for e in test_doc:
            #a = model.docvecs.most_similar(positive=[model.infer_vector(e)],topn=5)
            b = model.infer_vector(e)
            #tableau_benigne.append(b)
            for i in range(0, len(b)-1):
                tableau_benigne.append(b[i])
        
        tableau_malware_final.append(tableau_benigne)
        b=[]
        tableau_benigne = []
    print(len(tableau_malware_final))
    return tableau_malware_final
   
def zip_benigne():
    a = 1
    data_benigne = []
    tableau_benigne_final = benigne()
    tab = [a for i in range(0, len(tableau_benigne_final))]
    data_benigne = [[i,j] for i,j in zip(tableau_benigne_final, tab)]
    #for element in zip(tableau_benigne_final, tab):
        #data_benigne.append(element)
    #data_benigne = np.array(data_benigne)
    #print("la longueur des benigne est :" .len(data_benigne))
    
    return data_benigne


def zip_malware():
    b=0
    tableau_malware_final = malware()
    data_malware = []
    tab = [b for i in range(0, len(tableau_malware_final))]
    data_malware = [[i,j] for i,j in zip(tableau_malware_final, tab)]
    #for element in zip(tableau_malware_final, tab):
        #data_malware.append(element)
    #data_malware = np.array(data_malware)
    #print("la longueur de data_malware est :".len(data_malware))
    return data_malware


def concat_to_tab():
    tab1 = zip_malware()
    tab2 = zip_benigne()
    #for element in tab1:
    tab2 += [element for element in tab1]
    random.shuffle(tab2)
    #print(tab2[20])
    #print(len(tab2))
    return tab2

def apk_cnn():
    x = []
    y = []
    data = np.array(concat_to_tab())
    batch_size = 100
    for element in data:
        #print(element[1])
        y.append(np.asarray(element[1]).astype(np.float32))
        x.append(np.asarray(element[0]).astype(np.float32))
    #x=np.array(x)
    #y=np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    
    for element in X_train:
        print(len(element))
        print(element)
    
    model = Sequential()
    model.add(layers.Conv1D(100,10,activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    history = model.fit(X_train, y_train , epochs=100, verbose=True, validation_data=(X_test,y_test), batch_size=batch_size)
    model.summary()
    
    print(history)
    loss, accuracy = model.evaluate(X_train , y_train , verbose=False)
    #print("************************Results for class :"+str(curr_class)+"*********************")
    print("Training Accuracy: {:.4f}".format(accuracy))
    print("ok")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    #loss, accuracy = model.evaluate(X_test_ones, Y_test_ones, verbose=False)
    #print("Testing Accuracy of class-1:  {:.4f}".format(accuracy))
    #loss, accuracy = model.evaluate(X_test_zeros, Y_test_zeros, verbose=False)
    #print("Testing Accuracy of class-0:  {:.4f}".format(accuracy))
    return (x,y)
if __name__ == "__main__":
    #benigne() 
    #malware() 
    #zip_benigne()
    #zip_malware()
    #concat_to_tab()
    apk_cnn()
    
