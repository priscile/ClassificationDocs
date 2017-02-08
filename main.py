'''
Created on 24 aout 2016

@author: priscile
'''
from __future__ import unicode_literals
from __future__ import division
#import os 
from nltk.corpus import PlaintextCorpusReader
#from operator import itemgetter
import os.path
import time
import pickle
from nltk.stem.snowball import SnowballStemmer


alphabet= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def mot_mot(alp,mt):
#    print mt
    while mt[-1] not in alp:
        mt=mt[0:len(mt)-1]      
#        print mt
        if mt == '':
            break
    return mt    

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
            SnowballStemmer("english").stem(mt)
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
            SnowballStemmer("english").stem(mt)    
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
    vect_doc = dictio
#    vect_doc = (dictio,classe_doc)
        
    return vect_doc


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


if __name__ == '__main__':
    
#liste des 10 meilleures categories (ayant plus de documents)
    categories_10 = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    vot='oui'

#je reccupere les chemins de mes donnees de train et test
    folder_path_train = "/home/priscile/dev/Reuters21578-Apte-115Cat/training" 
    folder_path_test = "/home/priscile/dev/Reuters21578-Apte-115Cat/test"

# Mon vocabulaire 
   
    
#mes features pour chaque classe
    
    fich1=open("store_features.cd") 
    all_dictio=pickle.load(fich1)
    for c in categories_10:
        print all_dictio[c]
    
    fich2=open("store_modeles.cd")
    dict_models=pickle.load(fich2)
#    fich3=open("test.txt","w")
#    for c in categories_10:
#        fich3.write(c)
#        print dict_models[c]._weights
#        fich3.write(dict_models[c]._weights)
#    fich3.close()
    
    print '---------------------A partir de ce programme vous pouvez determiner la classe dun document et faire une evaluation du modele global par ailleur vous pouvez verifier si un document est dune classe ou non et faire egalement une evaluation--------------------------'
    print '---classes valides---:', categories_10
    print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
    choix=raw_input()
    while choix not in ['1','2']:
        print '--------------entrez 1 pour determiner la classe dun document et 2  pour verifier si un document est dune classe ou non---------------'
        choix=raw_input()
    
    if choix == '1': 
        
        print '---construction du corpus de test---'
        deb = time.clock()     
        corpus_test_OVA=doc_test(folder_path_test, all_dictio, categories_10)
        print corpus_test_OVA
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
                for files in textlist.fileids()[:30]:
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
            for dirs in os.walk(folder_path_test):
                corpus_root = dirs[0] #parcour l'arborescence du chemin        
                if corpus_root != folder_path_test and os.path.basename(corpus_root) in categories_10:
                    classe_doc = os.path.basename(corpus_root)
                    textlist = PlaintextCorpusReader(corpus_root,'.*')
                    for files in textlist.fileids()[:30]:
                        test= corpus_root + '/' + files       
                        dict_result_classif=classify_one_doc(test, dict_models, categories_10)
                        classe_predite=OVA(dict_result_classif, categories_10)   
                        if classe_predite == classe_positive:
                            print 'classe: yes'
                        else:
                            print 'classe:No'    
#            print 'la classe predite pour le document est la suivante:',classe_doc
#            print 'label attendu:' ,l
                        if classe_doc==classe_positive  and classe_predite == classe_doc:
                            TP += 1
                        if classe_doc!=classe_positive  and classe_predite == classe_doc:
                            TN += 1    
            
                        if classe_doc==classe_positive and classe_predite != classe_doc :
                            FN += 1
                        if classe_doc!=classe_positive and classe_predite == classe_positive:
                            FP += 1        
#            print 'verification du label predit, true sil correspond et false si non:',correct
#            if correct=='true':
#                cmpt_correct += 1
            print  'Performance de la classe', classe_positive , ':', pre_rec(TP, FN, FP), '%'
            
            fin = time.clock()
            print '---Duree :',fin-deb,'secondes' 
            print 'Voulez vous tester une autre classe? Repondez par oui ou non'
            vot=raw_input()
            
            