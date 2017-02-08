'''
Created on 25 juin 2016

@author: priscile
'''

#!/sw/bin/python

import math
import sys
import glob
import pickle
import optimize
import numpy
from dicts import DefaultDict
from random import shuffle

Num=numpy

# In the documentation and variable names below "class" is the same
# as "category"

def train_maxent(train_instances, classes, gaussian_prior_variance):
    """Train and return a MaxEnt classifier.  
    The datastructure returned is dictionary whose keys are
    ('classname','word') tuples.  The values in the dictionary are
    the parameters (lambda weights) of the classifier.
    Note that this method does not return the list of classnames, 
    but the caller has those available already, since it is exactly the
    'classes' argument.  

    If you need to recover the classnames from the dictionary itself, 
    you'd need to do something like:
    maxent = train_maxent(instances, classes, variance)
    classes = list(set([c for (c,v) in maxent.keys()]))

    Some typical usage:
    classes = ['spam','ham']
    maxent = train_maxent(make_instances(classes), classes, 1.0)
    # interested in seeing the weight of "nigerian" in the "spam" class?
    lambda_spam_nigerian = maxent[('spam','nigerian')]
    # to classify some documents in directories corresponding to classes:
    for label, doc, name in make_instances(dirs):
        scores = classify_doc(maxent, classes, doc, name)
    """


    maxent = DefaultDict(0) 


    for cls in classes:
        maxent[(cls,'DEFAULT')] = 0
        for label, doc, name in train_instances:
            for word, v in doc.iteritems():
                maxent[(cls, word)] = 0

    # Remember the maxent features, and get the starting point for optimization
    features = maxent.keys()
    lambda0 = [0] * len(features)
    # Here call an optimizer to find the best lambdas
    lambdaopt = optimize.fminNCG(value, lambda0, gradient, args=(features, classes, train_instances, gaussian_prior_variance), printmessg=1, maxiter=5, avextol=1e-4)
    # Put the final optimal parameters are in returned dictionary
    assert maxent.keys() == features # Make sure the keys have not changed order
    maxent2 = DefaultDict(0)
    for k, v in zip(features, lambdaopt):
        maxent2[k] = v
    return maxent2

def make_instances(dirs):
    classes = dirs
    instances = []
    for cls in classes:
        for fichier in glob.glob(cls+"/*"):
            doc = DefaultDict(0)
            name = fichier
            for word in open(fichier).read().split():
                word = word.lower()
                doc[word] = 1
            instances.append((cls, doc, name))
    return instances

def gradient(lambdas, features, classes, instances, gaussian_prior_variance):
    feature_count = len(lambdas)
    grad = Num.zeros(feature_count) # optimize expects this to be a numpy array

    # TO DO: implement the gradient of the likelihood function (including prior)
    # remember to return the negative gradient because fminNCG minimizes.

    return grad

def value(lambdas, features, classes, instances, gaussian_prior_variance):
    """Return the log-likelihood of the true labels
    of the instances, using the parameters given in lambdas, where those lambdas
    correspond to the (word,class) keys given in 'features'."""
    # Build a MaxEnt classifier dictionary from the keys and lambdas
    maxent = dict([(k,v) for (k,v) in zip(features, lambdas)])
    # Use this MaxEnt classifier to classify all the documents in dirs
    # Accumulate the log-likelihood of the correct class
    total_log_prob = 0
    for label, doc, name in instances:
        class_probs = classify_doc(maxent, classes, doc, name)
        classes_to_probs = dict([(cls, prob) for prob, cls in class_probs])
        true_class_prob = classes_to_probs[label]
        total_log_prob += math.log(true_class_prob)

    prior_log_prob = 0.0
    # TO DO: Incorporate a Gaussian prior on parameters here!

    print "value:"
    print (total_log_prob + prior_log_prob)    

    # Return the NEGATIVE total_log_prob because fminNCG minimizes, 
    # and we want to MAXIMIZE log probability
    return -(total_log_prob + prior_log_prob)

def classify_doc(maxent, classes, doc, name):
    """Given a trained MaxEnt classifier returned by train_maxent(), and
    a test document tuple, doc, return an array of tuples, each
    containing a class label and the probability of the class according
    to the classifier."""
    scores = []
    ##
    # print 'Classifying', name
    for c in classes:
        ##
        #Put in the weight for the default feature
        score = maxent[(c, 'DEFAULT')]
        ##
        #Put in the weight for all the words in the document
        for word, v in doc.iteritems():
            weight = maxent[(c, word)]
            score += weight * v
        scores.append(score)
    # exp() and normalize the scores to turn them into probabilities
    maximum = max(scores)
    scores = [math.exp(x - maximum) for x in scores]
    normalizer = sum(scores)
    scores = [x / normalizer for x in scores]
    # make the scores list actually contain tuples like (0.84, "spam")
    scores = zip(scores, classes)
    return scores

def test_classifier(maxent, classes, instances, set_name):
    if len(instances) == 0:
        return
    correct = 0.0
    for label, doc, name in instances:
        ##
        # print "========="
        ##
        # print "true label: " + label
        res = classify_doc(maxent, dirs, doc, name)
        cls = max(res, key = lambda x:  x[0])
        ##
        # print "pred label: " + cls
        if cls == label:
            correct += 1.0
    print set_name + " accuracy: "
    print (correct / len(instances))

if __name__ == '__main__':
    print 'argv', sys.argv
    print "Usage:", sys.argv[0], "classdir1 classdir2 [classdir3...] train_portion"
    dirs = sys.argv[1:-1]
    train_portion = sys.argv[-1]
    #########
    dirs = ["spamham/easy_ham_2", "spamham/spam_2"]
    classes = dirs
    #########
    train_portion = 0.7
    gaussian_prior_variance = 10.0

    instances = make_instances(dirs)
    shuffle(instances)

    num_train = int(round(float(train_portion)) * len(instances))

    train_instances = instances[:num_train]
    test_instances = instances[num_train:]

    maxent = train_maxent (train_instances, classes, gaussian_prior_variance)

    test_classifier(maxent, classes, train_instances, "Train set")
    test_classifier(maxent, classes, test_instances, "Test set")

    pickle.dump(maxent, open("maxent.pickle", 'w'))

# E.g. type at command line
# python maxent.py spam ham 0.7
# where 0.7 is the portion of the data to use to train the model,
# and the model's accuracy will be evaluated on the other unseen portion.
# You will need the Numpy library to be installed.
# Otherwise you can implement your own conjugate gradient method, 
# which isn't very hard either.  For example, see "Numeric Recipes in C".