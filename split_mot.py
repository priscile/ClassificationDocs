'''
Created on 8 sept. 2016

@author: priscile
'''
from __future__ import unicode_literals
from __future__ import division
#import os 
#import nltk
#import re
#from nltk import wordpunct_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
#from operator import itemgetter
import os.path
import time
#import pickle
#from nltk import stem
#from nltk.stem.snowball import SnowballStemmer
#!from numpy import maximum



folder_path_train = "/home/priscile/dev/Reuters21578-Apte-115Cat/training" 
folder_path_test = "/home/priscile/dev/Reuters21578-Apte-115Cat/test"
cachedStopWords = stopwords.words("english")
alphabet= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
categories_10 = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]

def liste_class(corpus):
    liste_classes=[]
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus:
            liste_classes.append(os.path.basename(corpus_root))
    
    return liste_classes

def mot_mot(alp,mt):
#    print mt
    while mt[-1] not in alp:
        mt=mt[0:len(mt)-1]      
#        print mt
        if mt == '':
            break
    return mt  


def mot_mot_1(alp,mt1):
    i=0
    best=''
    mt=mot_mot(alp, mt1)
#    print mt
    while i<len(mt) and mt[i] in alphabet:
        i=i+1
    if i==len(mt):
        best=mt
#        print best    
    return best 

############# liste les mot d'un document----------------
def split(doc):
    min_length=3
    tokens=[]
    fs = open(doc,'r') 
    for ligne in fs:
        l=ligne.split() 
#        print l               
        for chaine in l:
            if len(chaine)>= min_length:
                chaine=chaine.lower()
#                print chaine
                chaine=mot_mot_1(alphabet, chaine)
                if chaine!='' and chaine not in cachedStopWords:
                    tokens.append(chaine)
                    
    return tokens

##############compte les occurences des mots dans un document deja sous forme de liste---------------
def freq_mot(tokens):
    dic_doc={}
    for mot in tokens:
        if mot not in dic_doc:
            dic_doc[mot]=0
        dic_doc[mot]=dic_doc[mot]+1
    
    return dic_doc

#text="/home/priscile/test"
#textlist = PlaintextCorpusReader(text,'.*')
#for files in textlist.fileids():
#    doc= text + '/' + files
#    l=split(doc)
#    print l 
#    print freq_mot(l)
    
##############donne la frequence des mots d'une classe-----------------------    
def dict_freq_mot_classe(classe,corpus):
    dictio={}
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus:
            if os.path.basename(corpus_root)==classe:
                textlist = PlaintextCorpusReader(corpus_root,'.*')
                for files in textlist.fileids():
                    test= corpus_root + '/' + files
                    l=split(test)
                    d=freq_mot(l)
                    for mot in d:
                        if mot not in dictio:
                            dictio[mot]=0
                        dictio[mot]=dictio[mot]+d[mot]
                        
    return dictio

############dictionnaire des classes, chaque classe contenant un dictionnaire des frequences de ses mots
def dict_freq_all_classe(categories, corpus):
    dictio_all={} 
    for c in categories:
        print '---construction des frequences des mots de la classe---',c
        deb = time.clock()
        dictio_all[c]=dict_freq_mot_classe(c, corpus)
        fin = time.clock()
        print '---Duree :',fin-deb,'secondes'
    
    
    return dictio_all

ensemble_classes=liste_class(folder_path_train)

print '---construction des dictionnaires de mots de chaque classe---' 
debut = time.clock() 
dic=dict_freq_all_classe(ensemble_classes, folder_path_train)
final = time.clock()
print '---Duree :',final-debut,'secondes'

#############construction du vocabulaire-------------
def vocabulaire(dict_all_classe):
    vocab=[]
    for c in dict_all_classe:
        for mot in dict_all_classe[c]:
            if mot not in vocab:
                vocab.append(mot)
                    
    return vocab    

print '---construction du vocabulaire---' 
debut = time.clock() 
voc=vocabulaire(dic)
final = time.clock()
print '---Duree :',final-debut,'secondes'
#print '-------longueur du vocabulaire:', len(voc) 


def taille_corpus(corpus):
    taille = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != corpus:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                taille += os.path.getsize(test)
                
    return taille 

#####frequece d'un mot dans une classe
def freq_mot_classe(mot,classe,dictio):
    freq=0
    dic_class=dictio[classe]
    if mot in dic_class:
        freq=dic_class[mot]
    
    return freq 


###########frequence d'un mot dans les classes differents d'une classe donnee
def freq_mot_classe_else(mot,classe,dictio):
    freq=0
    for c in dictio:
        if c!=classe:
            freq=freq+freq_mot_classe(mot, c, dictio)

    return freq 


##########nombre de mot different d'un mot donne dans une classe
def nbr_mot_else(mot,classe,dictio):
    
    if mot in dictio[classe]:       
        return len(dictio[classe])-1 
    else:
        return len(dictio[classe])

#########nombre de mot different d'un mot donne dans les classes differente d'un classe donnee
def nbr_mot(mot,classe,dictio):
    nbr=0
    for c in dictio:
        if c!=classe:
            nbr=nbr+nbr_mot_else(mot, c, dictio)
                    
    return nbr 


###############fonction de calcul du X-square tel que presente dans l'article de yannis et al sur ME###################
def x_square(n,a,c,e,g):
    
    if ((a+c)*(a+e)*(c+g)*(e+g)) == 0:
        X=-1
    else:
        X=(n*((a*g - c*e)**2))/((a+c)*(a+e)*(c+g)*(e+g))
    #((a+c)*(a+e)*(c+g)*(e+g))
    
    return X 


#fonction qui calcule l'ensemble des features d'une classe donnee
def features_glob(corpus,vocab,dictio,categories):
    N=taille_corpus(corpus)
    dict_cat={}
    for classe in categories:
        dict_cat[classe]={}
    for mot in vocab:
        for classe in categories:
            A=freq_mot_classe(mot, classe, dictio)
            B=freq_mot_classe_else(mot, classe, dictio)
            C=nbr_mot_else(mot, classe, dictio)
            D=nbr_mot(mot, classe, dictio)
            X=x_square(N, A, B, C, D)
            dict_cat[classe][mot]=X      
    return dict_cat



print '---construction du vecteur de feature pour la classe crude-----' 
debut = time.clock() 
print features_glob(folder_path_train, voc[:100], dic,categories_10)
final = time.clock()
print '---Duree :',final-debut,'secondes'


