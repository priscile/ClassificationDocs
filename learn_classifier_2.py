'''
Created on 31 aout 2016

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




########verification  d'un mot alphabetique ##################""""

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

###fonction qui compte le nombre de mots du corpus#########

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


##############fonction qui compte les occurences d'un mot dans une classe######""'
def occurrence_mot_i_classe(mot,classe,corpus):
    compteur = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if os.path.basename(corpus_root) == classe:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                x = open(test,'r')
                for ligne in x:
                    lign=ligne.split()
                    for mt in lign:
                        mt=mot_mot(alphabet, mt)
                        if mt==mot:                       
#                    if ligne.find(mot)>0:
                            compteur+=1
                x.close()
    return compteur

##############fonction qui compte les occurences des racine de mots dans une classe######""'
def occurrence_mot_i_classe_stem(mot,classe,corpus):
    compteur = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if os.path.basename(corpus_root) == classe:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
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
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                x = open(test,'r')
                for ligne in x:
                    lign=ligne.split()
                    for mt in lign:
                        mt=mot_mot(alphabet, mt)
                        if mt==mot:  
#                    if ligne.find(mot)>0:
                            compteur+=1
                x.close()
    return compteur

######"####"fonction qui compte les occurrences d'un mot racinise dans toutes les classes(dans le corpus)#####################"
def occurrence_mot_i_corpus_stem(mot,corpus):
    compteur = 0
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        if corpus_root != corpus:
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
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
            for files in textlist.fileids():
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

def features_glob_stem(corpus,classe,vocab):
    N=taille_corpus(corpus)
    D=taille_classe(classe, corpus)
    caracteristiques={}
    for mot in vocab:
        A=occurrence_mot_i_classe_stem(mot, classe, corpus)
        B=occurrence_mot_i_corpus_stem(mot, corpus)
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
        for t in f[:2000]:
            for m in e[t]:
                dictio_s[m]=t
        print dictio_s    
        all_dictio[cat]=dictio_s
#        
    return all_dictio

def dict_dict_features_stem(corpus,categories,vocab):
    
    all_dictio={}
   
    for cat in categories:
        dictio= features_glob_stem(corpus, cat, vocab)
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
        for t in f[:2000]:
            for m in e[t]:
                dictio_s[m]=t
        print dictio_s    
        all_dictio[cat]=dictio_s
#        
    return all_dictio

###############fonction qui represente un document sous forme vectorielle  ######################

def represent_docs(corpus,cat,dictio_classes,categories):
    
    docs_train = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        
        if corpus_root != corpus:
            if os.path.basename(corpus_root) == cat:
                dictio = dictio_classes[cat]
                textlist = PlaintextCorpusReader(corpus_root,'.*')
                for files in textlist.fileids():
                    lign=[]
                    test= corpus_root + '/' + files
                    x = open(test,'r')
                    for ligne in x:
                        lign=lign+ligne.split()
                    for mt in lign:
                        if len(mt)>=3:
                            mt=mot_mot(alphabet, mt) 
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
                    for files in textlist.fileids():
                        lign=[]
                        test= corpus_root + '/' + files
                        x = open(test,'r')
                        for ligne in x:
                            lign=lign+ligne.split()
                        for mt in lign:
                            if len(mt)>=3:
                                mt=mot_mot(alphabet, mt)
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



###############fonction qui represente un document sous forme vectorielle version steemming ######################


def represent_docs_stem(corpus,cat,dictio_classes,categories):
    
    docs_train = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin
        
        if corpus_root != corpus:
            if os.path.basename(corpus_root) == cat:
                dictio = dictio_classes[cat]
                textlist = PlaintextCorpusReader(corpus_root,'.*')
                for files in textlist.fileids():
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
                    for files in textlist.fileids():
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
    lign_1=[]
    x = open(doc,'r')
    for ligne in x:
        lign=lign+ligne.split()
    for mt in lign:
        if len(mt)>3:
            mt=mot_mot(alphabet, mt)
            lign_1.append(mt) 
#    l=dictio.items()
#    l.sort(key=itemgetter(1),reverse=True)
#    l=l[:30]
#    l=dict(l)  
    print lign_1             
    for mot,fval in dictio.items():
        val=fval
        if mot in lign_1:
#            if ligne.find(mot)>0:
            dictio[mot]=val
        else:
            dictio[mot]=0.0                          
    x.close()
    vect_doc = (dictio,classe_doc)
        
    return vect_doc


##########"""""""fonction qui represente sous forme vectorielle les documents de test version stemming #########################
def represent_doc_test_stem(doc,classe_doc,dictio_classes):
    
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


def train_models_stem(corpus,dictio_classes,categories,cat_model):
         
    documents_train=represent_docs_stem(corpus,cat_model,dictio_classes,categories)
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
        
    
def apprentissage_des_models_stem(corpus,dictio_classe,categories):
    dict_models = {}
    for cat in categories:
        classifier=train_models_stem(corpus, dictio_classe, categories, cat)
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
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['acq']=doc_classe
#            print dict_result_classif['acq']
        elif cat == 'corn' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['corn']=doc_classe
#            print dict_result_classif['corn']
        elif cat == 'crude':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['crude']=doc_classe
#            print dict_result_classif['crude']
        elif cat == 'earn':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['earn']=doc_classe  
#            print dict_result_classif['earn']          
        elif cat == 'grain' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['grain']=doc_classe
#            print dict_result_classif['grain']
        elif cat == 'interest':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['interest']=doc_classe
#            print dict_result_classif['interest']
        elif cat == 'money-fx' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['money-fx']=doc_classe
#            print dict_result_classif['money-fx']
        elif cat == 'ship' :
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)               
            dict_result_classif['ship']=doc_classe
#            print dict_result_classif['ship']
        elif cat == 'trade':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['trade']=doc_classe
#            print dict_result_classif['trade']
        elif cat == 'wheat':
            classifier=dictio_models[cat]
            results = classifier.classify(doc)
            print results
            if results=='Yes':
                doc_classe = (doc, [cat])
            else:
                final_cl=[]
                for cl in categories:
                    if cl != cat:
                        final_cl.append(cl)
                doc_classe = (doc,final_cl)
            dict_result_classif['wheat']=doc_classe
#            print dict_result_classif['wheat']

    return dict_result_classif


############""fonction qui represente l'ensemble des document de test #####################
def doc_test(corpus,dictio_feature,categories):
    docs_test = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus and os.path.basename(corpus_root) in categories:
            classe_doc = os.path.basename(corpus_root)
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                doc_cat = represent_doc_test(test, classe_doc, all_dictio)
                docs_test.append(doc_cat)

    return docs_test


def doc_test_stem(corpus,dictio_feature,categories):
    docs_test = []
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus and os.path.basename(corpus_root) in categories:
            classe_doc = os.path.basename(corpus_root)
            textlist = PlaintextCorpusReader(corpus_root,'.*')
            for files in textlist.fileids():
                test= corpus_root + '/' + files
                doc_cat = represent_doc_test_stem(test, classe_doc, all_dictio)
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


if __name__ == '__main__':
    
#liste des 10 meilleures categories (ayant plus de documents)
    categories_10 = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]
    alphabet= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    vot='oui'

#je reccupere les chemins de mes donnees de train et test
    folder_path_train = "/home/priscile/dev/Reuters21578-Apte-115Cat/training" 
    folder_path_test = "/home/priscile/dev/Reuters21578-Apte-115Cat/test"
    
    cachedStopWords = stopwords.words("english")
    
    
    print '---------------------A partir de ce programme vous pouvez  construire un modele dapprentissage base sur la tache de stemming et un autre nutilisant pas cette tache la et vous pouvez tester les modeles--------------------------'
    print '----------NB: lors du test des modeles, lon peut le faire a nimporte quel moment sans avoir besoin dapprendre a nouveau les modeles si on les a deja appris une fois----------------'
    print '----------------------Entrez 1 pour apprendre les modeles et 2 pour tester---------------------'
    ch=raw_input()
    while ch not in ['1','2']:
        print '-------------Entrez 1 pour apprendre les modeles et 2 pour tester---------------'
        ch=raw_input() 
    if ch == '1': 
        print '--------------entrez 1 pour lapprentisage dun modele simple et 2  pour lapprentissage dun modele avec stemming---------------'
        choix_0=raw_input()  
        while choix_0 not in ['1','2']:
            print '-------------entrez 1 pour lapprentisage dun modele simple et 2  pour lapprentissage dun modele avec stemming---------------'
            choix_0=raw_input()  
        
        if choix_0 == '1':
        
            print '---construction du vocabulaire---'
            deb = time.clock()
            vocabulaire= tokenisation(folder_path_train) 
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'  
            print 'la taille du vocabulaire est:', len(vocabulaire) 
            print vocabulaire[:10]

        
            print '---construction des vecteurs de features pour chaque classe---'
            deb = time.clock()  
            f=open("store_features.cd","w")  
            all_dictio = dict_dict_features(folder_path_train, categories_10, vocabulaire)
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
             
        elif choix_0 == '2':
            
            print '---construction du vocabulaire raciniser---'
            deb = time.clock()
            vocabulaire_stem= stemming(folder_path_train) 
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'  
            print 'la taille du vocabulaire est:', len(vocabulaire_stem) 
            print vocabulaire_stem[:10]
   
            print '---construction des vecteurs de features pour chaque classe---'
            deb = time.clock()  
            f=open("store_features_stem.cd","w")  
            all_dictio = dict_dict_features_stem(folder_path_train, categories_10, vocabulaire)
            pickle.dump(all_dictio,f)
            f.close()
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'
            
            print '---Apprentissage des models par classe---'
            deb = time.clock() 
            fich=open("store_modeles_stem.cd","w")    
            dict_models=apprentissage_des_models_stem(folder_path_train, all_dictio, categories_10)
            pickle.dump(dict_models,fich)
            fich.close()
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'
            
    elif  ch == '2': 
        print '----------------------Entrez 1 pour tester le model simple et 2 pour tester le modele avec stemming---------------------'
        ch_0=raw_input()
        while ch_0 not in ['1','2']:
            print '-------------Entrez 1 pour apprendre les modeles et 2 pour tester---------------'
            ch_0=raw_input() 
        
        if ch_0 == '1': 
            fich1=open("store_features.cd") 
            all_dictio=pickle.load(fich1)
#            for c in categories_10:
#                print all_dictio[c]
    
            fich2=open("store_modeles.cd")
            dict_models=pickle.load(fich2)
        
            print '---------------------A partir de cette section vous pouvez determiner la classe dun document et faire une evaluation du modele global par ailleur vous pouvez verifier si un document est dune classe ou non et faire egalement une evaluation--------------------------'
            print '---classes valides---:', categories_10
            print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
            choix=raw_input()
            while choix not in ['1','2']:
                print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
                choix=raw_input()
                
            if choix == '1': 
        
                print '---construction du corpus de test---'
                deb = time.clock()     
                corpus_test_OVA=doc_test(folder_path_train, all_dictio, categories_10)
#                print corpus_test_OVA
                fin = time.clock()
                print '---Duree :',fin-deb,'secondes'
                print 'le nombre de documents a tester est le suivant:',len(corpus_test_OVA)
                cmpt_correct=0
                print '---debut du processus de test---'
                deb = time.clock()    
                for (fs,l) in corpus_test_OVA:
#                    print '---classification---'    
                    dict_result_classif=classify_one_doc(fs, dict_models, categories_10)
                    classe_doc=OVA(dict_result_classif, categories_10)
#                    print dict_result_classif
                    print 'la classe predite pour le document est la suivante:',classe_doc
                    print 'label attendu:' ,l
                    if classe_doc == l:
                        correct='true'
                    else:
                        correct='false'    
                    print 'verification du label predit, true sil correspond et false si non:',correct
                    if correct=='true':
                        cmpt_correct += 1
            
                fin = time.clock()
                print '---Duree :',fin-deb,'secondes'    
                
                print '--accuracy--:',accuracy(cmpt_correct, len(corpus_test_OVA)), '%'       

            elif choix == '2':
                while vot=='oui':
                    print 'entrez la classe que vous designez comme classe positive'
                    classe_positive=raw_input()
                    while classe_positive not in categories_10:
                        print 'error: la classe est non valide,entrer a nouveau la classe '
                        classe_positive=raw_input()
                    print '---construction du corpus de test---'
                    deb = time.clock()     
                    corpus_test_OVA=doc_test(folder_path_test, all_dictio, categories_10)
                    fin = time.clock()
                    print '---Duree :',fin-deb,'secondes'
                    print 'le nombre de documents a tester est le suivant:',len(corpus_test_OVA)
                    cmpt_correct=0
                    print '---debut du processus de test---'
                    deb = time.clock()    
                    for (fs,l) in corpus_test_OVA:
#                        print '---classification---'    
                        dict_result_classif=classify_one_doc(fs, dict_models, categories_10)
                        classe_doc=OVA(dict_result_classif, categories_10)
#                        print classe_doc
#                        if classe_doc == classe_positive:
#                            print 'classe: yes'
#                        else:
#                            print 'classe:No'    
#            print 'la classe predite pour le document est la suivante:',classe_doc
#            print 'label attendu:' ,l
                        if l==classe_positive  and classe_doc == l:
                            TP += 1
                        if l!=classe_positive  and classe_doc == l:
                            TN += 1    
            
                        if l==classe_positive and classe_doc != l :
                            FN += 1
                        if l!=classe_positive and classe_doc == classe_positive:
                            FP += 1        
#            print 'verification du label predit, true sil correspond et false si non:',correct
#            if correct=='true':
#                cmpt_correct += 1
                    print  'Performance de la classe', classe_positive , ':', pre_rec(TP, FN, FP), '%'
            
                    fin = time.clock()
                    print '---Duree :',fin-deb,'secondes' 
                    print 'Voulez vous tester une autre classe? Repondez par oui ou non'
                    vot=raw_input()
               
                
                
        elif  ch_0 == '2': 
            fich1=open("store_features_stem.cd") 
            all_dictio=pickle.load(fich1)
            for c in categories_10:
                print all_dictio[c]
    
            fich2=open("store_modeles_stem.cd")
            dict_models=pickle.load(fich2)  
            
            print '---------------------A partir de cette section vous pouvez determiner la classe dun document et faire une evaluation du modele global par ailleur vous pouvez verifier si un document est dune classe ou non et faire egalement une evaluation--------------------------'
            print '---classes valides---:', categories_10
            print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
            choix=raw_input()
            while choix not in ['1','2']:
                print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
                choix=raw_input()
                
            if choix == '1': 
        
                print '---construction du corpus de test---'
                deb = time.clock()     
                corpus_test_OVA=doc_test_stem(folder_path_test, all_dictio, categories_10)
                print corpus_test_OVA
                fin = time.clock()
                print '---Duree :',fin-deb,'secondes'
                print 'le nombre de documents a tester est le suivant:',len(corpus_test_OVA)
                cmpt_correct=0
                print '---debut du processus de test---'
                deb = time.clock()    
                for (fs,l) in corpus_test_OVA:
                    print '---classification---'    
                    dict_result_classif=classify_one_doc(fs, dict_models, categories_10)
                    classe_doc=OVA(dict_result_classif, categories_10)
                    print 'la classe predite pour le document est la suivante:',classe_doc
                    print 'label attendu:' ,l
                    if classe_doc == l:
                        correct='true'
                    else:
                        correct='false'    
                    print 'verification du label predit, true sil correspond et false si non:',correct
                    if correct=='true':
                        cmpt_correct += 1
            
                fin = time.clock()
                print '---Duree :',fin-deb,'secondes'    
                
                print '--accuracy--:',accuracy(cmpt_correct, len(corpus_test_OVA)), '%'       

            elif choix == '2':
                while vot=='oui':
                    print 'entrez la classe que vous designez comme classe positive'
                    classe_positive=raw_input()
                    while classe_positive not in categories_10:
                        print 'error: la classe est non valide,entrer a nouveau la classe '
                        classe_positive=raw_input()
                    print '---construction du corpus de test---'
                    deb = time.clock()     
                    corpus_test_OVA=doc_test_stem(folder_path_test, all_dictio, categories_10)
                    fin = time.clock()
                    print '---Duree :',fin-deb,'secondes'
                    print 'le nombre de documents a tester est le suivant:',len(corpus_test_OVA)
                    cmpt_correct=0
                    print '---debut du processus de test---'
                    deb = time.clock()    
                    for (fs,l) in corpus_test_OVA:
                        print '---classification---'    
                        dict_result_classif=classify_one_doc(fs, dict_models, categories_10)
                        classe_doc=OVA(dict_result_classif, categories_10)
                        if classe_doc == classe_positive:
                            print 'classe: yes'
                        else:
                            print 'classe:No'    
#            print 'la classe predite pour le document est la suivante:',classe_doc
#            print 'label attendu:' ,l
                        if l==classe_positive  and classe_doc == l:
                            TP += 1
                        if l!=classe_positive  and classe_doc == l:
                            TN += 1    
            
                        if l==classe_positive and classe_doc != l :
                            FN += 1
                        if l!=classe_positive and classe_doc == classe_positive:
                            FP += 1        
#            print 'verification du label predit, true sil correspond et false si non:',correct
#            if correct=='true':
#                cmpt_correct += 1
                    print  'Performance de la classe', classe_positive , ':', pre_rec(TP, FN, FP), '%'
            
                    fin = time.clock()
                    print '---Duree :',fin-deb,'secondes' 
                    print 'Voulez vous tester une autre classe? Repondez par oui ou non'
                    vot=raw_input()                