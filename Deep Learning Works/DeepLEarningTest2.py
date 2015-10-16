from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import nolearnOld
from nolearnOld import dbn
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from csv import DictReader
from collections import defaultdict
from numpy import array
from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.datasets import load_digits


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

import databaseCodes.mongoOperations as mongoOperations
from collections import defaultdict
import re
from sklearn import linear_model


from textblob import TextBlob
import nltk
from nltk import bigrams
from nltk import trigrams
from lasagne.layers.noise import dropout


negWordList = []


def populateNegWord():
    """total_data = DictReader(open("swearWords.csv", 'Ur'))
    for ii in total_data:
        print ii
        negWordList.append(str(ii['words']))
    negWordList.append("fuck")"""
    
    total_data = open("swearWords.csv", 'r')
    
    for line in total_data:
        words = line.split(",")
        for word in words:
            negWordList.append(word)
    
    print negWordList
    
    
def getBiGramsFromComments(text):
    # split the texts into tokens
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if len(token) > 1] #same as unigrams
    bi_tokens = bigrams(tokens)
    fdist = nltk.FreqDist(bi_tokens)
    return fdist

def getTriGramsFromComments(text):
    # split the texts into tokens
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if len(token) > 1] #same as unigrams
    tri_tokens = trigrams(tokens)
    fdist = nltk.FreqDist(tri_tokens)
    return fdist

def getUnigramsFromComments(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def loadEmotionDictionary():
    total_data = DictReader(open("video_categorization.csv", 'Ur'))
    emotionDictionary = defaultdict(str)
    for data in total_data:
        emotion = str(data["question2"])
        videolink = str(data["videolink"])
        emotionDictionary[videolink] = emotion
    return emotionDictionary

def loadContentDictionary():
    total_data = DictReader(open("video_categorization.csv", 'Ur'))
    emotionDictionary = defaultdict(str)
    for data in total_data:
        emotion = str(data["question3"])
        videolink = str(data["videolink"])
        emotionDictionary[videolink] = emotion
    return emotionDictionary




def FeatureExtractionVersionTwo(data,emotionDictionary,contentdictionary):
    featureDictionary = defaultdict(int)
    
    videolink = data["videolink"]
    
    #user features
    featureDictionary["userFollowerCount"] = int(data["userFollowerCount"])
    featureDictionary["userFollowingCount"] = int(data["userFollowingCount"])
    featureDictionary["userloopCount"] = int(data["userloopCount"])
    featureDictionary["userAuthoredPostCountCount"] = int(data["userAuthoredPostCountCount"])
    featureDictionary["userLikeCount"] = int(data["userLikeCount"])
    featureDictionary["userPostCount"] = int(data["userPostCount"])
    featureDictionary["userDescriptionPolarity"] =float(1.0- float(data["userDescriptionPolarity"]))
    featureDictionary["userDescriptionSubjectivity"] = float(1.0- float(data["userDescriptionSubjectivity"]))
    
    
    """#media session features
    
    
    featureDictionary["postLikeCount"] = int(data["postLikeCount"])
    featureDictionary["postLoopCount"] = int(data["postLoopCount"])
    featureDictionary["postCommentCount"] = int(data["postCommentCount"])
    featureDictionary["postRepostCount"] = int(data["postRepostCount"])
    featureDictionary["postExplicitContent"] = int(data["postExplicitContent"])
    featureDictionary["postDescriptionPolarity"] = float(1.0- float(data["postDescriptionPolarity"]))
    featureDictionary["postDescriptionSubjectivity"] = float(1.0- float(data["postDescriptionSubjectivity"]))
        
    #comment features
    
    featureDictionary["tagCount"] = int(data["tagCount"])
    featureDictionary["mentionCount"] = int(data["mentionCount"])
    featureDictionary["otherCount"] = int(data["otherCount"])
    featureDictionary["verifiedCount"] = int(data["verifiedCount"])
    featureDictionary["nonVerifiedCount"] = int(data["nonVerifiedCount"])
    
    featureDictionary["ownerCommentCount"] = int(data["ownerCommentCount"])
    featureDictionary["ownerCommentPolarityTotal"] = 100.0 + float(data["ownerCommentPolarityTotal"])
    featureDictionary["ownerCommentSubjectivityTotal"] = 100.0 + float(data["ownerCommentSubjectivityTotal"])
    
    
    featureDictionary["otherCommentCount"] = int(data["otherCommentCount"])
    featureDictionary["otherCommentPolarityTotal"] = float(100.0 + float(data["otherCommentPolarityTotal"]))
    featureDictionary["otherCommentSubjectivityTotal"] = float(100.0 + float(data["otherCommentSubjectivityTotal"]))
    
    featureDictionary["allCommentPolarityTotal"] = float(100.0 + float(data["allCommentPolarityTotal"]))
    featureDictionary["allCommentSubjectivityTotal"] = float(100.0 + float(data["allCommentSubjectivityTotal"]))
    
    featureDictionary["negativeCommentCount"] = int(float(data["negativeCommentCount"]))
    featureDictionary["negativeWordCount"] = int(data["negativeWordCount"])
    featureDictionary["negativePolarityTotal"] = float(100.0 + float(data["negativePolarityTotal"]))
    featureDictionary["negativeSubjectivityTotal"] = float(100.0 +float(data["negativeSubjectivityTotal"]))
    featureDictionary["negativeCommentPercentage"] = float(data["negativeCommentPercentage"])
    featureDictionary["negativeWordPerNegativeComment"] = float(data["negativeWordPerNegativeComment"])"""
    

    
           

    return featureDictionary
    

def featureSelectionProcess(X,Y,featureSelection):
    print "feature selection process: "+str(featureSelection)
    print "before feature selection. shape of X"+str(X[0].shape)
    if featureSelection == "linearSVM":
        X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, Y)
        print  "after feature selection. shape of X"+str(X_new[0].shape)
    elif featureSelection == "SelectKBest":
        X_new = SelectKBest(chi2, k=6).fit_transform(X, Y)
        print  "after feature selection. shape of X"+str(X_new[0].shape)
    elif featureSelection == "SelectKPercentile":
        selector = SelectPercentile(f_classif, percentile=30)
        X_new = selector.fit_transform(X, Y)
        print  "after feature selection. shape of X"+str(X_new[0].shape)
    elif featureSelection == "TreeBased":
        clf = ExtraTreesClassifier()
        X_new = clf.fit(X, Y).transform(X)
        print  "after feature selection. shape of X"+str(X_new[0].shape)
    elif featureSelection == "Recursive":
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
        X_new = rfe.fit(X, Y).transform(X)
        print  "after feature selection. shape of X"+str(X_new[0].shape)


def SplitIntoTestTrainingDataset(filename,emotionDictionary,contentdictionary):
    total_data = DictReader(open(filename, 'Ur'))
    bullyingCount = 0
    notBullyngCount = 0
    
    trainingData = []
    trainingLabel = []
    testData = []
    testLabel = []
    count = 0
    for ii in total_data:
        print count
        count = count + 1
        if float(ii["question2:confidence"]) < 0.6:
            continue
        else:
            if ii["question2"] == "noneBll":
                label = 0
                notBullyngCount = notBullyngCount + 1
                if notBullyngCount > 258: #258
                    continue
                if notBullyngCount > 100:   #100
                    featureDictionary = FeatureExtractionVersionTwo(ii,emotionDictionary,contentdictionary)
                    testData.append(featureDictionary)
                    testLabel.append(label)
                else:
                    featureDictionary = FeatureExtractionVersionTwo(ii,emotionDictionary,contentdictionary)
                    trainingData.append(featureDictionary)
                    trainingLabel.append(label)

            else:
                label = 1
                bullyingCount = bullyingCount + 1
                if bullyingCount > 179:   #179
                    continue
                if bullyingCount > 100:  #100
                    featureDictionary = FeatureExtractionVersionTwo(ii,emotionDictionary,contentdictionary)
                    testData.append(featureDictionary)
                    testLabel.append(label)
                else:
                    featureDictionary = FeatureExtractionVersionTwo(ii,emotionDictionary,contentdictionary)
                    trainingData.append(featureDictionary)
                    trainingLabel.append(label)
            if bullyingCount > 179 and notBullyngCount > 258:
                break
    print str(len(testData))+" test data size"
    print str(len(testLabel))+" test label size"
    
    print str(len(trainingData))+" training data size"
    print str(len(trainingLabel))+" training label size"
    
    totalData = []
    totalLabel = []
    
    for label in trainingLabel:
        totalLabel.append(label)
    for label in testLabel:
        totalLabel.append(label)
    
    for data in trainingData:
        totalData.append(data)
    for data in testData:
        totalData.append(data)
        
    print "total data collected "+str(len(totalLabel))
    print "total label collected "+str(len(totalData))
    

    return(totalData,totalLabel,199)


def TransformIntoVectors(totalData,totalLabel):
    v = DictVectorizer(sparse=True)
    
    X =  v.fit_transform(totalData)   
    Y = array(totalLabel)
    
    return (X,Y)
    

def confidenceLogRegression(theta, X,intercept):
    h = X.dot(theta.T)+intercept
    m = h.size-1
    return h[m] 
    
        
    
    

emotionDictionary = loadEmotionDictionary()
contentdictionary =  loadContentDictionary()

populateNegWord()

print "=========================================================="
totalData,totalLabel,index = SplitIntoTestTrainingDataset("vine_meta_data.csv",emotionDictionary,contentdictionary)
print "getting data is finished successfully"
print "=========================================================="





print "=========================================================="
print "now transforming data into vectorized form"
X,Y = TransformIntoVectors(totalData,totalLabel)
print "data has been convereted into vectorized form successfully"
print "=========================================================="



def TransformIntoMatrix(data,target):
    row = data.shape[0]
    column = data.shape[1]
    m = np.empty((row, column))

    for i in range(row):
        k = 0
        for j in data[i].indices:
            m[i][j] = data[i].data[k]
            k = k+1
            
    # Prepend the column of 1s for bias
    N, M  = m.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = m
    
    return (all_X, target)

X_training = X[0:200]
Y_training = Y[0:200]
X_test = X[200:437]
Y_test = Y[200:437]


X_Matrix,Y_Matrix = TransformIntoMatrix(X,Y)
X_Matrix = X_Matrix / X_Matrix.max(axis=0)

X_training_Matrix = X_Matrix[0:200]
Y_training_Matrix = Y_Matrix[0:200]
X_test_Matrix = X_Matrix[200:437]
Y_test_Matrix = Y_Matrix[200:437]



########################################################################
net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden1', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=X_training_Matrix.shape,
        hidden1_num_units=1000,  # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=2,  

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.1,
        update_momentum=0.9,
        max_epochs=500,
        verbose=1,
        )
net1.fit(X_training_Matrix,Y_training_Matrix)

logReg = linear_model.LogisticRegression(C=1e6)
logReg.fit(X_training_Matrix, Y_training_Matrix)
dbn = dbn.DBN(
    [X_training.shape[1],1000,2],
    learn_rates = 0.3,
    learn_rate_decays = 0.9,
    epochs = 10,
    verbose = 1)
dbn.fit(X_training, Y_training)

net1Preds,net1Probs = net1.predict(X_test_Matrix)
dbnPreds,dbnProbs = dbn.predict(X_test)
logRegPreds = logReg.predict(X_test_Matrix)

print "neural network"
print classification_report(Y_test_Matrix, net1Preds)
print "deep belief network"
print classification_report(Y_test, dbnPreds)
print "logistic regression"
print classification_report(Y_test_Matrix, logRegPreds)

count = 0
i = 0
while i < 237:
    if Y_test[i] == net1Preds[i] == logRegPreds[i] == dbnPreds[i]:
        count = count + 1
        print 
    i = i+1

print count
print "done"


"""
########################################################################


logReg = linear_model.LogisticRegression(C=1e6)
logReg.fit(X_training, Y_training)

########################################################################

dbn = dbn.DBN(
    [X_training.shape[1],1000,2],
    learn_rates = 0.3,
    learn_rate_decays = 0.9,
    epochs = 10,
    verbose = 1)
dbn.fit(X_training, Y_training)

########################################################################
preds,probs = dbn.predict(X_test)
i=0
correct = 0 
total = 0
for xt in X_test:
    #xt = np.reshape(xt, (1, xt.shape[0]))
    dbn_pred,dbn_prob = dbn.predict(xt)
    theta = logReg.coef_[0]
    intercept =  logReg.intercept_
    logReg_pred = confidenceLogRegression(theta, xt,intercept)
    if float(dbn_prob[0][1]) >= 0.55:
        if dbn_pred[0] == Y_test[i]:
            correct +=1
        preds[i] = 1
        total+=1 
        #print "__________________________"
        
    elif float(dbn_prob[0][0]) >= 0.53:
        if dbn_pred[0] == Y_test[i]:
            correct +=1
        total+=1
        preds[i] = 0 
        
    elif (float(logReg_pred) - 6.0) > 0:
        prediction = 1
        if prediction == Y_test[i]:
            correct +=1
        total+=1
        preds[i] = 1
    elif (float(logReg_pred) < - 1.5):
        prediction = 0
        if prediction == Y_test[i]:
            correct +=1
        total+=1
        preds[i] = 0
    else:
        if float(abs(net_probs[i][0]-net_probs[i][1])) > 0.70 and net_preds[i] == preds[i]:
            preds[i] = net_preds[i]
            if preds[i] == Y_test[i]:
                correct +=1
            total+=1
        if net_probs[i][0] < 0.50:
            preds[i] = 1
    i+=1
    
    

print str(total)+" total"
print str(correct)+" correct"

print classification_report(Y_test, preds)"""

print "done"

