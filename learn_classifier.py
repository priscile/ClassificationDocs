'''
Created on 8 juil. 2016

@author: priscile
'''

from __future__ import unicode_literals
from __future__ import division
#import os 
import nltk
import re
#from nltk import wordpunct_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
#from operator import itemgetter
import os.path
import time
import pickle
#from nltk import stem
from nltk.stem.snowball import SnowballStemmer
#!from numpy import maximum

cachedStopWords = stopwords.words("english")

folder_path_train = "/home/priscile/dev/Reuters21578-Apte-115Cat/training" 
folder_path_test = "/home/priscile/dev/Reuters21578-Apte-115Cat/test"

categories_10 = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]
alphabet= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#stemmer=stem.PorterStemmer()

def mot_mot(alp,mt):
#    print mt
    while mt[-1] not in alp:
        mt=mt[0:len(mt)-1]      
#        print mt
        if mt == '':
            break
    return mt    
    
#fonction de tokenisation qui prend en parametre un chemin vers un dossier(train ou test)
#representant un corpus afin de produire un vocabulaire
def tokenisation (path):
    tokens = []
    min_length =3
    for dirs in os.walk(path):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != path:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                fs = open(test,'r')
                for ligne in fs:
                    l=ligne.split()                   
                    for chaine in l:
                        if len(chaine)>= min_length:
                            chaine=chaine.lower()
                            chaine=mot_mot(alphabet, chaine)
                            p = re.compile('[a-zA-Z]+')
                            if chaine not in cachedStopWords and p.match(chaine):
                                tokens.append(chaine)
#    vocab = []
#    for words in tokens:
#        vocab.append(SnowballStemmer("english").stem(words))
    
    tokens1=set(tokens)
    tokens2=list(tokens1)                    
#              
    return tokens2

####token_stemmed####
def stemming (path):
    tokens = []
    min_length =3
    for dirs in os.walk(path):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != path:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                fs = open(test,'r')
                for ligne in fs:
                    l=ligne.split()                   
                    for chaine in l:
                        if len(chaine)>= min_length:
                            chaine=chaine.lower()
                            chaine=mot_mot(alphabet, chaine)
                            p = re.compile('[a-zA-Z]+')
                            if chaine not in cachedStopWords and p.match(chaine):
                                tokens.append(chaine)
    vocab = []
    for words in tokens:
        vocab.append(SnowballStemmer("english").stem(words))
    
    tokens1=set(vocab)
    tokens2=list(tokens1)                    
#              
    return tokens2




print '---construction du vocabulaire---'
deb = time.clock()
vocabulaire= tokenisation(folder_path_train) 
fin = time.clock()
print '---Duree :',fin-deb,'secondes'  
print 'la taille du vocabulaire est:', len(vocabulaire) 
print vocabulaire[:10]


print '---construction du vocabulaire raciniser---'
deb = time.clock()
vocabulaire_stem= stemming(folder_path_train) 
fin = time.clock()
print '---Duree :',fin-deb,'secondes'  
print 'la taille du vocabulaire est:', len(vocabulaire_stem) 
print vocabulaire_stem[:10]
####################selection des features pour le calcul des fonctions caracteristiques #######################

###fonction qui compte le nombre de mots du corpus#########

def taille_corpus(corpus):
    taille = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != corpus:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids()[:30]:
                test= corpus_root + '/' + files
                taille += os.path.getsize(test)
                
    return taille 


##############fonction qui compte les occurences d'un mot dans une classe######""'
def occurrence_mot_i_classe(mot,classe,corpus):
    compteur = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if os.path.basename(corpus_root) == classe:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids()[:30]:
                test= corpus_root + '/' + files
                x = open(test,'r')
                for ligne in x:
                    lign=ligne.split()
                    for mt in lign:
                        mt=mot_mot(alphabet, mt)
                        mt=SnowballStemmer("english").stem(mt)
                        if mt==mot:                       
#                    if ligne.find(mot)>0:
                            compteur+=1
                x.close()
    return compteur


######"####"fonction qui compte les occurrences d'un mot dans toutes les classes(dans le corpus)#####################"
def occurrence_mot_i_corpus(mot,corpus):
    compteur = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != corpus:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids()[:30]:
                test= corpus_root + '/' + files
                x = open(test,'r')
                for ligne in x:
                    lign=ligne.split()
                    for mt in lign:
                        mt=mot_mot(alphabet, mt)
                        mt=SnowballStemmer("english").stem(mt)
                        if mt==mot:  
#                    if ligne.find(mot)>0:
                            compteur+=1
                x.close()
    return compteur


################"fonction qui compte le nombre de mot d'une classe #################
def taille_classe(classe,corpus):
    taille = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if os.path.basename(corpus_root) == classe:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids()[:30]:
                test= corpus_root + '/' + files
                taille += os.path.getsize(test)
                
    return taille

###############fonction de calcul du X-square tel que presente dans l'article de yannis et al sur ME###################
def x_square(n,a,c,e,g):
    
    if ((a+c)*(a+e)*(c+g)*(e+g)) == 0:
        X=-1
    else:
        X=(n*((a*g - c*e)**2))/((a+c)*(a+e)*(c+g)*(e+g))
    #((a+c)*(a+e)*(c+g)*(e+g))
    
    return X 

#fonction qui calcule l'ensemble des features d'une classe donnee
def features_glob(corpus,classe,vocab):
    N=taille_corpus(corpus)
    D=taille_classe(classe, corpus)
    caracteristiques={}
    for mot in vocab:
        A=occurrence_mot_i_classe(mot, classe, corpus)
        B=occurrence_mot_i_corpus(mot, corpus)
        C=B-A
        E=D-A 
        F=N-D
        G=F-C 
        X=x_square(N, A, C, E, G)
        caracteristiques[mot]=X
        
    return caracteristiques

#fonction qui retourne un dictionnaire de dictionnaire de features de chaque classes
def dict_dict_features(corpus,categories,vocab):
    
    all_dictio={}
   
    for cat in categories:
        dictio= features_glob(corpus, cat, vocab)
#        print dictio
        e={}
        for k in dictio:
            if dictio[k] not in e:
                e[dictio[k]]=[]
            e[dictio[k]].append(k)  
        f=sorted(e)
        f.reverse()
        print f        
        dictio_s={}
        for t in f[:5]:
            for m in e[t]:
                dictio_s[m]=t
        print dictio_s    
        all_dictio[cat]=dictio_s
#        
    return all_dictio


###############fonction qui represente un document sous forme vectorielle ######################
def represent_doc_train(doc,classe_doc,learn_cat,dictio_classes):
    lign=[]
    if classe_doc==learn_cat:
        dictio = dictio_classes[learn_cat]
        x = open(doc,'r')
        for ligne in x:
            lign=lign+ligne.split()
        for mt in lign:
            if len(mt)>= 3:                
                mt=mot_mot(alphabet, mt)
            mt=SnowballStemmer("english").stem(mt)    
#        l=dictio.items()
#        l.sort(key=itemgetter(1),reverse=True)
#        l=l[:30]
#        l=dict(l)               
        for mot,fval in dictio.items():
            val=fval
            if mot in lign:
#                if ligne.find(mot)>0:
                dictio[mot]=val
            else:
                dictio[mot]=0.0                          
        x.close()
        vect_doc = (dictio,'Yes')
    else:
        x = open(doc,'r')
        for ligne in x:
            lign=lign+ligne.split()
        for mt in lign:
            if len(mt)>=3:
                mt=mot_mot(alphabet, mt) 
            mt=SnowballStemmer("english").stem(mt)   
        dictio = dictio_classes[classe_doc]
#        l=dictio.items()
#        l.sort(key=itemgetter(1),reverse=True)
#        l=l[:30]
#        l=dict(l)               
        for mot,fval in dictio.items():
            val_1=fval
            if mot in lign:
#                if ligne.find(mot)>0:
                dictio[mot]=val_1
            else:
                dictio[mot]=0.0                        
        x.close()
        vect_doc = (dictio,'No')  
              
    return vect_doc 

def represent_docs(corpus,cat,dictio_classes,categories):
    
    docs_train = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        
        if corpus_root != corpus:
            if os.path.basename(corpus_root) == cat:
                dictio = dictio_classes[cat]
                textlist = PlaintextCorpusReader(corpus_root,'.*')
                for files in textlist.fileids()[:30]:
                    lign=[]
                    test= corpus_root + '/' + files
                    x = open(test,'r')
                    for ligne in x:
                        lign=lign+ligne.split()
                    for mt in lign:
                        if len(mt)>=3:
                            mt=mot_mot(alphabet, mt) 
                        mt=SnowballStemmer("english").stem(mt)   
#                    l=dictio.items()
#                    l.sort(key=itemgetter(1),reverse=True)
#                    l=l[:30]
#                    l=dict(l)               
                    for (mot,fval) in dictio.items():
                        val=fval
                        if mot in lign:
#                            if ligne.find(mot)>0:
                            dictio[mot]=val
                        else:
                            dictio[mot]=0.0                     
                    x.close()
                    docs_train.append((dictio,'Yes'))
            else:
                if os.path.basename(corpus_root) in categories:
                    cat_else = os.path.basename(corpus_root)
                    dictio = dictio_classes[cat_else]
                    textlist = PlaintextCorpusReader(corpus_root,'.*')
                    for files in textlist.fileids()[:30]:
                        lign=[]
                        test= corpus_root + '/' + files
                        x = open(test,'r')
                        for ligne in x:
                            lign=lign+ligne.split()
                        for mt in lign:
                            if len(mt)>=3:
                                mt=mot_mot(alphabet, mt)
                            mt=SnowballStemmer("english").stem(mt)
#                        l=dictio.items()
#                        l.sort(key=itemgetter(1),reverse=True)
#                        l=l[:30]
#                        l=dict(l)               
                        for (mot,fval) in dictio.items():
                            val_1=fval
                            if mot in lign:                                    
#                                if ligne.find(mot)>0:
                                dictio[mot]=val_1
                            else:
                                dictio[mot]=0.0                 
                        x.close()
                        docs_train.append((dictio,'No'))        
    return docs_train


##########"""""""fonction qui represente sous forme vectorielle les documents de test #########################
def represent_doc_test(doc,classe_doc,dictio_classes):
    
    dictio = dictio_classes[classe_doc]
    lign=[]
    x = open(doc,'r')
    for ligne in x:
        lign=lign+ligne.split()
    for mt in lign:
        if len(mt)>=3:
            mt=mot_mot(alphabet, mt) 
        mt=SnowballStemmer("english").stem(mt)   
#    l=dictio.items()
#    l.sort(key=itemgetter(1),reverse=True)
#    l=l[:30]
#    l=dict(l)               
    for mot,fval in dictio.items():
        val=fval
        if mot in lign:
#            if ligne.find(mot)>0:
            dictio[mot]=val
        else:
            dictio[mot]=0.0                          
    x.close()
    vect_doc = (dictio,classe_doc)
        
    return vect_doc

###########################fonction qui retourne le classifieur d'une classe au cas ou il voudrait etre teste en solo ####################
def train_models(corpus,dictio_classes,categories,cat_model):
         
    documents_train=represent_docs(corpus,cat_model,dictio_classes,categories)
    classifier = nltk.classify.MaxentClassifier.train(documents_train,labels=['Yes','No'])
                
    return classifier

def test_model (model,document):
    
    results = model.classify(document)
    
    return results
    
#evaluation d'un model
def precision_recall(corpus,model):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for (fs, l) in corpus:
        r=test_model(model, fs)
        if l == 'Yes' and r == l :
            TP += 1
        elif l == 'No' and r == l :
            TN += 1
            
        if l == 'Yes' and r != l :
            FN += 1
        elif l == 'No' and r != l :
            FP += 1        
        
    precision = TP/(TP+FP)
    
    recall = TP/(TP+FN)
    
#    f_mesure = (2*precision*recall)/(precision+recall)
   
    return ((2*precision*recall)/(recall+precision))*100.0

################fonction qui retoure un dictionnaire de models############
def apprentissage_des_models(corpus,dictio_classe,categories):
    dict_models = {}
    for cat in categories:
        classifier=train_models(corpus, dictio_classe, categories, cat)
        dict_models[cat]=classifier
        
    return dict_models    
        
    


###################focntion de classification dd'un document par un model########################
#def classify_one_doc(corpus_1,doc ,dictio_classes,categories):
def classify_one_doc(doc ,dictio_models,categories):
    dict_result_classif={}
    for cat in categories:
        if cat == 'acq':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['acq']=doc_classe
        elif cat == 'corn' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['corn']=doc_classe
        elif cat == 'crude':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['crude']=doc_classe
        elif cat == 'earn':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['earn']=doc_classe            
        elif cat == 'grain' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['grain']=doc_classe
        elif cat == 'interest':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['interest']=doc_classe
        elif cat == 'money-fx' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['money-fx']=doc_classe
        elif cat == 'ship' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)               
            dict_result_classif['ship']=doc_classe
        elif cat == 'trade':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['trade']=doc_classe
        elif cat == 'wheat':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['wheat']=doc_classe

    return dict_result_classif

############""fonction qui represente l'ensemble des document de test #####################
def doc_test(corpus,dictio_feature,categories):
    docs_test = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus and os.path.basename(corpus_root) in categories:
            classe_doc = os.path.basename(corpus_root)
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids()[:30]:
                test= corpus_root + '/' + files
                doc_cat = represent_doc_test(test, classe_doc, all_dictio)
                docs_test.append(doc_cat)

    return docs_test



def OVA(dict_result_classif, categories):
    init=categories[0]
    dict_cpt={}
    cat=''
    for x in dict_result_classif[init]:
        classe=x         
    for cat in categories[1:len(categories)]:
        classe_else= dict_result_classif[cat]
        for y in classe_else:
            cl=y
        classe= set(classe) & set(cl)
        classe=list( classe)
        if len(classe)==1:
            cat=classe[0]
        else:
            for cls in categories:
                cpt = 0
                for liste in dict_result_classif:
                    if cls in liste:
                        cpt=cpt+1
                dict_cpt[cls]=cpt
            mx=0    
            for a,b in dict_cpt.items():
                if b>mx :
                    mx=b
                    cat=a
                             
                                     
                
    return cat     

def accuracy(nbr_bien_classe,taille_ens_test):
    return (nbr_bien_classe/taille_ens_test)*100.0

def pre_rec(TP,FN,FP):
    
   
    if (TP+FP)==0:
        precision=0
    else:
        precision = TP/(TP+FP)
            
     
    if (TP+FN)==0:
        recall=0
    else:  
        recall = TP/(TP+FN)
    
    return ((2*precision*recall)/(recall+precision))*100.0

print '---construction des vecteurs de features pour chaque classe---'
deb = time.clock()  
f=open("store_features.cd","w")  
all_dictio = dict_dict_features(folder_path_train, categories_10, vocabulaire[:50])
pickle.dump(all_dictio,f)
f.close()
fin = time.clock()
print '---Duree :',fin-deb,'secondes'


print '---Apprentissage des models par classe---'
deb = time.clock() 
fich=open("store_modeles.cd","w")    
dict_models=apprentissage_des_models(folder_path_train, all_dictio, categories_10)
pickle.dump(dict_models,fich)
fich.close()
fin = time.clock()
print '---Duree :',fin-deb,'secondes'



print '---version stemmer:construction des vecteurs de features pour chaque classe---'
deb = time.clock()  
f=open("store_features_stem.cd","w")  
all_dictio = dict_dict_features(folder_path_train, categories_10, vocabulaire_stem[:50])
pickle.dump(all_dictio,f)
f.close()
fin = time.clock()
print '---Duree :',fin-deb,'secondes'

print '---Apprentissage des models par classe---'
deb = time.clock() 
fich=open("store_modeles_stem.cd","w")    
dict_models=apprentissage_des_models(folder_path_train, all_dictio, categories_10)
pickle.dump(dict_models,fich)
fich.close()
fin = time.clock()
print '---Duree :',fin-deb,'secondes'



#Main--------

    