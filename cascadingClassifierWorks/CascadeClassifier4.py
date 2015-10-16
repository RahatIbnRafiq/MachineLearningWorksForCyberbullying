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
import math


negWordList = []


def populateNegWord():
    
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
    emotion = str(emotionDictionary[videolink])
    content = str(contentdictionary[videolink])
    
    #user features
    featureDictionary["userFollowerCount"] = int(data["userFollowerCount"])
    featureDictionary["userFollowingCount"] = int(data["userFollowingCount"])
    featureDictionary["userloopCount"] = int(data["userloopCount"])
    featureDictionary["userAuthoredPostCountCount"] = int(data["userAuthoredPostCountCount"])
    featureDictionary["userLikeCount"] = int(data["userLikeCount"])
    featureDictionary["userPostCount"] = int(data["userPostCount"])
    featureDictionary["userDescriptionPolarity"] =float(1.0- float(data["userDescriptionPolarity"]))
    featureDictionary["userDescriptionSubjectivity"] = float(1.0- float(data["userDescriptionSubjectivity"]))
    
    
    #media session features
    
    
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
    featureDictionary["negativeWordPerNegativeComment"] = float(data["negativeWordPerNegativeComment"])
    

    featureDictionary[emotion]+=1
    featureDictionary[content]+=1
           

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
    






emotionDictionary = loadEmotionDictionary()
contentdictionary =  loadContentDictionary()

populateNegWord()

print "=========================================================="
totalData,totalLabel,index = SplitIntoTestTrainingDataset("vine_meta_data.csv",emotionDictionary,contentdictionary)
print "getting data is finished successfully"
print "=========================================================="


def LogisticRegressionUserFeature(x_test,logReg_userfeature):
    x_test_user = x_test[:,0:8]#taking user features
    prediction = logReg_userfeature.predict(x_test_user)[0]
    theta = logReg_userfeature.coef_[0]
    intercept =  logReg_userfeature.intercept_
    confidence = abs(confidenceLogRegression(theta, x_test_user,intercept)-0)
    return (prediction,confidence)

def LogisticRegressionCommentFeature(x_test,logReg_commentfeature):
    x_test_comment = x_test[:,15:35]#taking user features
    prediction = logReg_commentfeature.predict(x_test_comment)[0]
    theta = logReg_commentfeature.coef_[0]
    intercept =  logReg_commentfeature.intercept_
    confidence = abs(confidenceLogRegression(theta, x_test_comment,intercept)-0)
    return (prediction,confidence)


print "=========================================================="
print "now transforming data into vectorized form"
X,Y = TransformIntoVectors(totalData,totalLabel)
print "data has been convereted into vectorized form successfully"
print "=========================================================="


X_Matrix,Y_Matrix = TransformIntoMatrix(X,Y)
X_Matrix = X_Matrix / X_Matrix.max(axis=0)

X_training_Matrix = X_Matrix[0:200]
Y_training_Matrix = Y_Matrix[0:200]
X_test_Matrix = X_Matrix[200:437]
Y_test_Matrix = Y_Matrix[200:437]



userFeatureTraining =  X_training_Matrix[0:200,0:8]
logReg_userfeature = linear_model.LogisticRegression(C=1e6)
logReg_userfeature.fit(userFeatureTraining, Y_training_Matrix)

mediaFeatureTraining =  X_training_Matrix[0:200,8:15]
logReg_mediafeature = linear_model.LogisticRegression(C=1e6)
logReg_mediafeature.fit(mediaFeatureTraining, Y_training_Matrix)

commentFeatureTraining =  X_training_Matrix[0:200,15:35]
logReg_commentfeature = linear_model.LogisticRegression(C=1e6)
logReg_commentfeature.fit(commentFeatureTraining, Y_training_Matrix)

commentFeatureTraining =  X_training_Matrix[0:200,15:35]
logReg_commentfeature = linear_model.LogisticRegression(C=1e6)
logReg_commentfeature.fit(commentFeatureTraining, Y_training_Matrix)

negativityFeatureTraining =  X_training_Matrix[0:200,31:35]
logReg_negativityfeature = linear_model.LogisticRegression(C=1e6)
logReg_negativityfeature.fit(negativityFeatureTraining, Y_training_Matrix)

allFeatureTraining =  X_training_Matrix[0:200,0:35]
logReg_allfeature = linear_model.LogisticRegression(C=1e6)
logReg_allfeature.fit(allFeatureTraining, Y_training_Matrix)


videoFeatureTraining =  X_training_Matrix[0:200,35:]
logReg_videofeature = linear_model.LogisticRegression(C=1e6)
logReg_videofeature.fit(videoFeatureTraining, Y_training_Matrix)


i = 0
total = 0
correct = 0
for x_test in X_test_Matrix:
    truth = Y_test_Matrix[i]
    x_test = np.reshape(x_test, (1, x_test.shape[0])) #reshaping the array into 2D
    prediction, confidence = LogisticRegressionUserFeature(x_test,logReg_userfeature)
    if confidence >=0.9 and prediction == 0:
        total = total + 1
        if prediction == truth:
            correct = correct + 1
    else:
        prediction, confidence = LogisticRegressionCommentFeature(x_test,logReg_commentfeature)
        if confidence >=1.0 and prediction == 0:
            total = total + 1
            if prediction == truth:
                correct = correct + 1
        
    i=i+1
print "total instances: "+str(total)
print "total correctly predicted instances: "+str(correct)
print "Calculated Accuracy is"+str(float(correct)/float(total)*100.0)

print "done"

