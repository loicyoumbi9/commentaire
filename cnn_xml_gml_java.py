from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import random
import itertools
from os import listdir
import os
#import re
import string
from glob import glob
from os.path import isfile, join, isdir
from pathlib import Path
from gensim.models import KeyedVectors
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


def nom_dossier(chemin):
    dossiers = [f for f in listdir(chemin) if isdir(join(chemin, f))]
    return dossiers

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
    if nb_ligne_para >= 1 :    
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
    tab_reste = []
    #print(len(tableau))
    #print("\n")
    
    return tableau
    
   


def benigne():
    tableau_benigne_final = []
    path = "benigne_deja_traite/"
    i = 0
    b = []
    a = []
    tableau_benigne = []
    manifest_tab = []
    #benigne_label =np.array([i for i in range(0,10)])

    model = Doc2Vec.load("epochs_50/para2vec.kv")
    a = nom_dossier(path)
    tableau = []
    for dossier in a:
        #b.append(nom_fichier(path+dossier+"/"))
        debut_chemin = path+dossier+"/"
        tableau_gml = []
        line_gml = open(debut_chemin+"gml.txt")
        for lines in line_gml.readlines():
       
            if(len(lines.split(' ')) > 3 ):
                for i in range (1, len(lines.split(' '))):
                    tableau_gml.append(lines.split(' ')[i])
                    
                tableau.append(tableau_gml)  
                tableau_gml = []
               
        
                    
        #p = ('.').join(element[0].split('.')[:-1])
            
        
        a = []
        line = open(path+dossier+"/"+dossier+".xml").readline()
        wv = KeyedVectors.load("vectors.kv", mmap='r')
        vector = [word.lower() for word in line.split(', ') if word != '']
                                             
        for word_a_traiter in vector:
            a = wv[word_a_traiter]
            tableau_benigne.append(np.array(a))
        #print("la longueur du tableau benigne est:",len(tableau_benigne))
        
        
        
        #tab_comp = []        
        #if len(tableau_benigne) <= 80 : 
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #    for i in range(len(tableau_benigne), 80):
        #        tableau_benigne.append(np.array(tab_comp))  
                
                
        #if (len(tableau) < 250):
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #    for i in range(len(tableau), 120):
        #        tableau.append(np.array(tab_comp))
                
                
        for t in tableau:
            tableau_benigne.append(np.array(t).astype(np.float32))
        
        
        test_doc = bloc_fichier_en_tableau(dossier+".txt", debut_chemin+'')
        tableau_benigne_java = []
        #a = model.docvecs.most_similar(positive=[model.infer_vector(e)],topn=5)
        for e in test_doc:
            b = model.infer_vector(e)
            #for i in range(0, len(b)-1):
            tableau_benigne_java.append(np.array(b))
        print("la longueur du tableau java est :",len(tableau_benigne_java))
        
        
        #if len(tableau_benigne_java) < 301:
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #    for i in range(len(tableau_benigne_java), 301):
        #        tableau_benigne_java.append(np.array(tab_comp)) 
                
        for t_java in tableau_benigne_java:
            tableau_benigne.append(np.array(t_java))
        print("lalongueur du tableau benigne des benigne avant completude est" ,len(tableau_benigne))
        if len(tableau_benigne) < 600:
            tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
            for i in range(len(tableau_benigne), 600):
                tableau_benigne.append(np.array(tab_comp)) 
        tableau_benigne_final.append(np.array(tableau_benigne))
        
        
        tableau_benigne_java = [] 
        tableau = []
        tableau_benigne = []
    for t in tableau_benigne_final:
        print(len(t))   
    return np.array(tableau_benigne_final)




def malware():
    tableau_benigne_final = []
    path = "malware_deja_traite/"
    i = 0
    b = []
    a = []
    tableau_benigne = []
    manifest_tab = []
    #benigne_label =np.array([i for i in range(0,10)])

    model = Doc2Vec.load("epochs_50/para2vec.kv")
    a = nom_dossier(path)
    tableau = []
    for dossier in a:
        #b.append(nom_fichier(path+dossier+"/"))
        debut_chemin = path+dossier+"/"
        tableau_gml = []
        line_gml = open(debut_chemin+"gml.txt")
        for lines in line_gml.readlines():
       
            if(len(lines.split(' ')) > 3 ):
                for i in range (1, len(lines.split(' '))):
                    tableau_gml.append(lines.split(' ')[i])
                    
                tableau.append(tableau_gml)  
                tableau_gml = []
               
        
                    
        #p = ('.').join(element[0].split('.')[:-1])
            
        
        a = []
        line = open(path+dossier+"/"+dossier+".xml").readline()
        wv = KeyedVectors.load("vectors.kv", mmap='r')
        vector = [word.lower() for word in line.split(', ') if word != '']
                                             
        for word_a_traiter in vector:
            a = wv[word_a_traiter]
            tableau_benigne.append(np.array(a))
        #print("la longueur du tableau benigne est:",len(tableau_benigne))
        
        
        
        #tab_comp = []        
        #if len(tableau_benigne) <= 80 : 
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #    for i in range(len(tableau_benigne), 80):
        #        tableau_benigne.append(np.array(tab_comp))  
                
                
        #if (len(tableau) < 250):
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #   for i in range(len(tableau), 120):
        #        tableau.append(np.array(tab_comp))
                
                
        for t in tableau:
            tableau_benigne.append(np.array(t).astype(np.float32))
        
        
        test_doc = bloc_fichier_en_tableau(dossier+".txt", debut_chemin+'')
        tableau_benigne_java = []
        #a = model.docvecs.most_similar(positive=[model.infer_vector(e)],topn=5)
        for e in test_doc:
            b = model.infer_vector(e)
            #for i in range(0, len(b)-1):
            tableau_benigne_java.append(np.array(b))
        print("la longueur du tableau java est :",len(tableau_benigne_java))
        
        
        #if len(tableau_benigne_java) < 301:
        #    tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
        #    for i in range(len(tableau_benigne_java), 301):
        #        tableau_benigne_java.append(np.array(tab_comp)) 
                
        for t_java in tableau_benigne_java:
            tableau_benigne.append(np.array(t_java))
        print("lalongueur du tableau benigne des malware avant completude est" ,len(tableau_benigne))
        if len(tableau_benigne) < 600:
            tab_comp =np.array([0 for i in range(0, 100)]).astype(np.float32)
            for i in range(len(tableau_benigne), 600):
                tableau_benigne.append(np.array(tab_comp)) 
        tableau_benigne_final.append(np.array(tableau_benigne))
        
        
        tableau_benigne_java = [] 
        tableau = []
        tableau_benigne = []
    for t in tableau_benigne_final:
        print(len(t)) 
      
        
    return tableau_benigne_final
            
    
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
    #print(data_benigne)
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
    batch_size = 2
    for element in data:
        if (len(element[0]) == 600) :
            #print(element[0])
            #print(len(element[0]))
            y.append(np.asarray(element[1]).astype(np.float32))
            x.append(np.asarray(element[0]).astype(np.float32))
    
    x=np.array(x)
    y=np.array(y)
    print(x)
    print(len(x))
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)
    print(len(X_train))
    model = Sequential()
    model.add(layers.Conv1D(filters=10, kernel_size=100,activation='relu'))
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
    
    return (x,y)
if __name__ == "__main__":
    #benigne() 
    #malware() 
    #zip_benigne()
    #zip_malware()
    #concat_to_tab()
    apk_cnn()
 


