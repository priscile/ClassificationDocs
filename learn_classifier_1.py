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


def split_stem(doc):
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
                    tokens.append(SnowballStemmer("english").stem(chaine))
                    
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

def dict_freq_mot_classe_stem(classe,corpus):
    dictio={}
    for dirs in os.walk(corpus):
        corpus_root = dirs[0] #parcour l'arborescence du chemin        
        if corpus_root != corpus:
            if os.path.basename(corpus_root)==classe:
                textlist = PlaintextCorpusReader(corpus_root,'.*')
                for files in textlist.fileids():
                    test= corpus_root + '/' + files
                    l=split_stem(test)
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


def dict_freq_all_classe_stem(categories, corpus):
    dictio_all={} 
    for c in categories:
        print '---construction des frequences des mots de la classe---',c
        deb = time.clock()
        dictio_all[c]=dict_freq_mot_classe_stem(c, corpus)
        fin = time.clock()
        print '---Duree :',fin-deb,'secondes'
    return dictio_all

#############construction du vocabulaire-------------
def vocabulaire(dict_all_classe):
    vocab=[]
    for c in dict_all_classe:
        for mot in dict_all_classe[c]:
            if mot not in vocab:
                vocab.append(mot)                
    return vocab    

def stemming(dict_all_classe):
    vocab=[]
    for c in dict_all_classe:
        for mot in dict_all_classe[c]:
            mot=SnowballStemmer("english").stem(mot)
            if mot not in vocab:
                vocab.append(mot)                
    return vocab 

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



#fonction qui retourne un dictionnaire de dictionnaire de features de chaque classes trier
def dict_dict_features(dict_cat,categories):
    
    all_dictio={}   
    for cat in categories:
        dictio= dict_cat[cat]
#        print dictio
        e={}
        for k in dictio:
            if dictio[k] not in e:
                e[dictio[k]]=[]
            e[dictio[k]].append(k)  
        f=sorted(e)
        f.reverse()
#        print f        
        dictio_s={}
        for t in f[:300]:
            for m in e[t]:
                dictio_s[m]=t
#        print dictio_s    
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
                    test= corpus_root + '/' + files
                    l=split(test)
#                    l=dictio.items()
#                    l.sort(key=itemgetter(1),reverse=True)
#                    l=l[:30]
#                    l=dict(l)               
                    for mot,fval in dictio.items():
                        val=fval
                        if mot in l:
#                            if ligne.find(mot)>0:
                            dictio[mot]=val
                        else:
                            dictio[mot]=0.0                     
                    docs_train.append((dictio,'Yes'))
            else:
                if os.path.basename(corpus_root) in categories:
                    cat_else = os.path.basename(corpus_root)
                    dictio = dictio_classes[cat_else]
                    textlist = PlaintextCorpusReader(corpus_root,'.*')
                    for files in textlist.fileids():
                        test= corpus_root + '/' + files
                        l=split(test)
#                        l=dictio.items()
#                        l.sort(key=itemgetter(1),reverse=True)
#                        l=l[:30]
#                        l=dict(l)               
                        for mot,fval in dictio.items():
#                            val_1=fval
                            if mot in l:                                    
#                                if ligne.find(mot)>0:
                                dictio[mot]=fval
                            else:
                                dictio[mot]=0.0                 
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
                    test= corpus_root + '/' + files
                    l=split_stem(test) 
#                    l=dictio.items()
#                    l.sort(key=itemgetter(1),reverse=True)
#                    l=l[:30]
#                    l=dict(l)               
                    for mot,fval in dictio.items():
                        val=fval
                        if mot in l:
#                            if ligne.find(mot)>0:
                            dictio[mot]=val
                        else:
                            dictio[mot]=0.0                     
                    docs_train.append((dictio,'Yes'))
            else:
                if os.path.basename(corpus_root) in categories:
                    cat_else = os.path.basename(corpus_root)
                    dictio = dictio_classes[cat_else]
                    textlist = PlaintextCorpusReader(corpus_root,'.*')
                    for files in textlist.fileids():
                        test= corpus_root + '/' + files
                        l=split_stem(test) 
#                        l=dictio.items()
#                        l.sort(key=itemgetter(1),reverse=True)
#                        l=l[:30]
#                        l=dict(l)               
                        for mot,fval in dictio.items():
                            val_1=fval
                            if mot in l:                                    
#                                if ligne.find(mot)>0:
                                dictio[mot]=val_1
                            else:
                                dictio[mot]=0.0                 
                        docs_train.append((dictio,'No'))        
    return docs_train




##########"""""""fonction qui represente sous forme vectorielle les documents de test #########################
def represent_doc_test(doc,classe_doc,dictio_classes):
    
    dictio = dictio_classes[classe_doc]
    l=split(doc)           
    for mot,fval in dictio.items():
        if mot in l:
#            if ligne.find(mot)>0:
            dictio[mot]=fval
        else:
            dictio[mot]=0.0                          
    
    
    vect_doc = dictio
#    vect_doc = (dictio,classe_doc)
        
    return vect_doc


##########"""""""fonction qui represente sous forme vectorielle les documents de test version stemming #########################
def represent_doc_test_stem(doc,classe_doc,dictio_classes):
    
    dictio = dictio_classes[classe_doc]
    l=split_stem(doc)      
    for mot,fval in dictio.items():
        val=fval
        if mot in l:
#            if ligne.find(mot)>0:
            dictio[mot]=val
        else:
            dictio[mot]=0.0                          
    vect_doc = (dictio,classe_doc)
        
    return vect_doc

###########################fonction qui retourne le classifieur d'une classe au cas ou il voudrait etre teste en solo ####################
def train_models(corpus,dictio_classes,categories,cat_model):
    
    print 'representation des documents dapprentissage'    
    documents_train=represent_docs(corpus,cat_model,dictio_classes,categories)
    print 'creation du modele'
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
def classify_one_doc(test ,dictio_models,categories):
    dict_result_classif={}
    for cat in categories:
        if cat == 'acq':
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['acq']
        elif cat == 'corn' :
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['corn']
        elif cat == 'crude':
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['crude']
        elif cat == 'earn':
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['earn']          
        elif cat == 'grain' :
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['grain']
        elif cat == 'interest':
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['interest']
        elif cat == 'money-fx' :
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['money-fx']
        elif cat == 'ship' :
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['ship']
        elif cat == 'trade':
            doc=represent_doc_test(test, cat, all_dictio)
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
#            print dict_result_classif['trade']
        elif cat == 'wheat':
            doc=represent_doc_test(test, cat, all_dictio)
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
            for files in textlist.fileids()[:1]:
                test= corpus_root + '/' + files
                doc_cat = represent_doc_test(test, classe_doc, all_dictio)
                print doc_cat
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
    ensemble_classes=liste_class(folder_path_train)
    
    
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
            print '---construction des dictionnaires de mots de chaque classe---' 
            debut = time.clock() 
            dic=dict_freq_all_classe(ensemble_classes, folder_path_train)
            final = time.clock()
            print '---Duree :',final-debut,'secondes'

        
            print '---construction du vocabulaire---'
            deb = time.clock()
            vocabulaire= vocabulaire(dic) 
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'  
            print 'la taille du vocabulaire est:', len(vocabulaire) 
#            print vocabulaire[:10]

        
            print '---construction des vecteurs de features pour toutes les classes---'
            deb = time.clock()  
            vecteur=features_glob(folder_path_train, vocabulaire, dic, categories_10)
            f=open("store_features.cd","w")  
            all_dictio = dict_dict_features(vecteur,categories_10)
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
            print '---construction des dictionnaires de mots de chaque classe---' 
            debut = time.clock() 
            dic=dict_freq_all_classe_stem(ensemble_classes, folder_path_train)
            final = time.clock()
            print '---Duree :',final-debut,'secondes'
            
            print '---construction du vocabulaire raciniser---'
            deb = time.clock()
            vocabulaire_stem= stemming(dic) 
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes'  
            print 'la taille du vocabulaire est:', len(vocabulaire_stem) 
            print vocabulaire_stem[:10]
   
            print '---construction des vecteurs de features pour chaque classe---'
            deb = time.clock()  
            f=open("store_features_stem.cd","w")  
            all_dictio = dict_dict_features(folder_path_train, categories_10, vocabulaire)
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
            while choix not in ['1','2','3']:
                print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
                choix=raw_input()
                
            if choix == '1': 
        
                print '---construction du corpus de test---'
                deb = time.clock()     
                corpus_test_OVA=doc_test(folder_path_test, all_dictio, categories_10)
#                print corpus_test_OVA
                fin = time.clock()
                print '---Duree :',fin-deb,'secondes'
                print 'le nombre de documents a tester est le suivant:',len(corpus_test_OVA)
                cmpt_correct=0
                print '---debut du processus de test---'
                deb = time.clock()    
                for dirs in os.walk(folder_path_test):
                    corpus_root = dirs[0] #parcour l'arborescence du chemin        
                    if corpus_root != folder_path_test and os.path.basename(corpus_root) in categories_10:
                        classe_doc = os.path.basename(corpus_root)
                        textlist = PlaintextCorpusReader(corpus_root,'.*')
                        for files in textlist.fileids()[:1]:
                            test= corpus_root + '/' + files    
#        for (fs,l) in corpus_test_OVA:
#            print '---classification---'    
                            dict_result_classif=classify_one_doc(test, dict_models, categories_10)
                            classe_predite=OVA(dict_result_classif, categories_10)
                            print 'la classe predite pour le document est la suivante:',classe_predite
                            print 'label attendu:' ,classe_doc
                            if classe_predite == classe_doc:
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
               
            elif choix== '3':
                while vot=='oui':
                    print 'entrez la classe que vous designez comme classe positive'
                    classe_positive=raw_input()
                    while classe_positive not in categories_10:
                        print 'error: la classe est non valide,entrer a nouveau la classe '
                        classe_positive=raw_input()
                    print '---debut du processus de test---'
                    deb = time.clock()    
                    for dirs in os.walk(folder_path_test):
                        corpus_root = dirs[0] #parcour l'arborescence du chemin        
                        if corpus_root != folder_path_test and os.path.basename(corpus_root) in categories_10:
                            l = os.path.basename(corpus_root)
                            textlist = PlaintextCorpusReader(corpus_root,'.*')
                            for files in textlist.fileids()[:1]:
                                test= corpus_root + '/' + files
                                doc=represent_doc_test(test, l, all_dictio)
                                print doc
                                classifier=dict_models[classe_positive]
                                results = classifier.classify(doc)
                                print results
                                if results=='Yes':
                                    classe_predite=classe_positive 
                                else:
                                    classe_predite=l
                                if l==classe_positive  and classe_predite == l:
                                    TP += 1
                                if l!=classe_positive  and classe_predite == l:
                                    TN += 1    
                                if l==classe_positive and classe_predite != l :
                                    FN += 1
                                if l!=classe_positive and classe_predite == classe_positive:
                                    FP += 1    
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
