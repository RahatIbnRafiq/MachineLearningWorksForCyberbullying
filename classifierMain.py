from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import LinearSVC
from textblob import TextBlob
import nltk
from nltk import bigrams
from nltk import trigrams

import csv
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE


from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_extraction import DictVectorizer
from csv import DictReader, DictWriter


from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

import databaseCodes.mongoOperations as mongoOperations
from collections import defaultdict
import re
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn import linear_model

import random
import warnings
import time
warnings.filterwarnings("ignore")

featureCount = 18
totalsamples = 358
negWordList = []


def populateNegWord():
    total_data = DictReader(open("new_neg_list1.csv", 'Ur'))
    for ii in total_data:
        negWordList.append(str(ii['words']))
    negWordList.append("fuck")

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

def FeatureExtractionVersionTwo(data,emotionDictionary,contentdictionary):
    featureDictionary = defaultdict(int)
    
    videolink = data["videolink"]
    
    
    emotion = str(emotionDictionary[videolink])
    content = str(contentdictionary[videolink])
    
    #user features
    #featureDictionary["userFollowerCount"] = int(data["userFollowerCount"])
    #featureDictionary["userFollowingCount"] = int(data["userFollowingCount"])
    #featureDictionary["userloopCount"] = int(data["userloopCount"])
    #featureDictionary["userAuthoredPostCountCount"] = int(data["userAuthoredPostCountCount"])
    #featureDictionary["userLikeCount"] = int(data["userLikeCount"])
    #featureDictionary["userPostCount"] = int(data["userPostCount"])
    #featureDictionary["userDescriptionPolarity"] =float(1.0- float(data["userDescriptionPolarity"]))
    #featureDictionary["userDescriptionSubjectivity"] = float(1.0- float(data["userDescriptionSubjectivity"]))
    
    
    #media session features
    
    
    #featureDictionary["postLikeCount"] = int(data["postLikeCount"])
    #featureDictionary["postLoopCount"] = int(data["postLoopCount"])
    #featureDictionary["postCommentCount"] = int(data["postCommentCount"])
    #featureDictionary["postRepostCount"] = int(data["postRepostCount"])
    #featureDictionary["postExplicitContent"] = int(data["postExplicitContent"])
    #featureDictionary["postDescriptionPolarity"] = float(1.0- float(data["postDescriptionPolarity"]))
    #featureDictionary["postDescriptionSubjectivity"] = float(1.0- float(data["postDescriptionSubjectivity"]))
    
    
    #comment features
    
    #featureDictionary["tagCount"] = int(data["tagCount"])
    #featureDictionary["mentionCount"] = int(data["mentionCount"])
    #featureDictionary["otherCount"] = int(data["otherCount"])
    #featureDictionary["verifiedCount"] = int(data["verifiedCount"])
    #featureDictionary["nonVerifiedCount"] = int(data["nonVerifiedCount"])
    
    #featureDictionary["ownerCommentCount"] = int(data["ownerCommentCount"])
    #featureDictionary["ownerCommentPolarityTotal"] = 100.0 + float(data["ownerCommentPolarityTotal"])
    #featureDictionary["ownerCommentSubjectivityTotal"] = 100.0 + float(data["ownerCommentSubjectivityTotal"])
    
    
    #featureDictionary["otherCommentCount"] = int(data["otherCommentCount"])
    #featureDictionary["otherCommentPolarityTotal"] = float(100.0 + float(data["otherCommentPolarityTotal"]))
    #featureDictionary["otherCommentSubjectivityTotal"] = float(100.0 + float(data["otherCommentSubjectivityTotal"]))
    
    #featureDictionary["allCommentPolarityTotal"] = float(100.0 + float(data["allCommentPolarityTotal"]))
    #featureDictionary["allCommentSubjectivityTotal"] = float(100.0 + float(data["allCommentSubjectivityTotal"]))
    
    #featureDictionary["negativeCommentCount"] = int(data["negativeCommentCount"])
    #featureDictionary["negativeWordCount"] = int(data["negativeWordCount"])
    #featureDictionary["negativePolarityTotal"] = float(100.0 + float(data["negativePolarityTotal"]))
    #featureDictionary["negativeSubjectivityTotal"] = float(100.0 +float(data["negativeSubjectivityTotal"]))
    featureDictionary["negativeCommentPercentage"] = float(data["negativeCommentPercentage"])
    featureDictionary["negativeWordPerNegativeComment"] = float(data["negativeWordPerNegativeComment"])
    
    
    """#negative word unigrams
    
    shareUrl = data["postShareUrl"]
    postDocument = mongoOperations.findPostsbyUrlFromCollection("VineDatabase", "SampledASONAMPosts", shareUrl)[0]
    postId = postDocument["postId"]
    commentDocuments = mongoOperations.findAllCommentsFromCollection("VineDatabase", "SampledASONAMComments", postId)
    for commentDocument in commentDocuments:
        commentText = str(commentDocument["commentText"].encode("utf8")).strip()
        #commentText =(commentText.decode('unicode_escape').encode('ascii','ignore'))
        if len(commentText.strip()) < 1:
            continue
        else:
            commentText =  commentText.lower()
            commentText = re.sub('[.!,@#$?-]', '', commentText)
            commentText = commentText.replace('.',' ')
            commentText = commentText.replace('^',' ')
            commentText = commentText.replace(',',' ')
            commentText = commentText.replace(';',' ')
            commentText = commentText.replace('[','')
            commentText = commentText.replace(']','')
            commentText = commentText.replace('[','')
            commentText = commentText.replace('}','')
            commentText = commentText.replace('{','')
            commentText = commentText.replace('``','')
            commentText = commentText.replace('"','')
            commentText = commentText.replace('``','')
            commentText = commentText.replace(')','')
            commentText = commentText.replace('(','')
            commentText = commentText.replace(':',' ')
            
            
            unigarms = getUnigramsFromComments(commentText)
            
            for unigram in unigarms:
                if unigram in ["fuck"]:
                    featureDictionary[unigram]+=1"""
    
    
    
    
    
    # codes for unigrams and bigrams
    
    
    """shareUrl = data["postShareUrl"]
    postDocument = mongoOperations.findPostsbyUrlFromCollection("VineDatabase", "SampledASONAMPosts", shareUrl)[0]
    postId = postDocument["postId"]
    commentDocuments = mongoOperations.findAllCommentsFromCollection("VineDatabase", "SampledASONAMComments", postId)
    for commentDocument in commentDocuments:
        commentText = str(commentDocument["commentText"].encode("utf8")).strip()
        #commentText =(commentText.decode('unicode_escape').encode('ascii','ignore'))
        if len(commentText.strip()) < 1:
            continue
        else:
            commentText =  commentText.lower()
            commentText = re.sub('[.!,@#$?-]', '', commentText)
            commentText = commentText.replace('.',' ')
            commentText = commentText.replace('^',' ')
            commentText = commentText.replace(',',' ')
            commentText = commentText.replace(';',' ')
            commentText = commentText.replace('[','')
            commentText = commentText.replace(']','')
            commentText = commentText.replace('[','')
            commentText = commentText.replace('}','')
            commentText = commentText.replace('{','')
            commentText = commentText.replace('``','')
            commentText = commentText.replace('"','')
            commentText = commentText.replace('``','')
            commentText = commentText.replace(')','')
            commentText = commentText.replace('(','')
            commentText = commentText.replace(':',' ')
            
            
            #trigram codes
            
            trigrams = getTriGramsFromComments(commentText)
            
            for k,v in trigrams.items():
                unit = str(k[0])+str(k[1])+str(k[2])  # trigrams
                featureDictionary[unit] += int(v)
                
                unit = str(k[0])   #unigrams
                featureDictionary[unit]+=1
                unit = str(k[1])
                featureDictionary[unit]+=1
                unit = str(k[2])
                featureDictionary[unit]+=1

            
            
            bigrams = getBiGramsFromComments(commentText)
            
            for k,v in bigrams.items():
                unit = str(k[0])+str(k[1])  # bigrams
                featureDictionary[unit] += int(v)
                
                unit = str(k[0])   #unigrams
                featureDictionary[unit]+=1
                unit = str(k[1])
                featureDictionary[unit]+=1
               
    featureDictionary[emotion]+=1
    featureDictionary[content]+=1"""
           

    return featureDictionary
    

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


def hybridClassifier(clf1,clf2,X,Y,X_test,Y_test):
    print "hybrid is now training"
    clf1.fit(X, Y)
    print "hybrid is now predicting"
    predictions1 = clf1.predict(X_test)
    
    clf2.fit(X, Y)
    predictions2 = clf2.predict(X_test)
    
    predictions = np.zeros(len(predictions1))
    
    j = 0
    while j < len(predictions):
        predictions[j] = predictions1[j]*predictions2[j]
        j = j+1
    
    accuracy = float(accuracy_score(Y_test, predictions, normalize=True))*100.0
    metricsValues =  precision_recall_fscore_support(Y_test, predictions, average='macro')
    precision = float(metricsValues[0])*100.0
    recall = float(metricsValues[1])*100.0
    
    print "the accuracy of this classifier is "+str(accuracy)
    print "the precision of this classifier is "+str(precision)
    print "the recall of this classifier is "+str(recall)
        

def classifierPerformance(classifier,X,Y,X_test,Y_test):
    print str(classifier[1])+" is now training"
    clf = classifier[0]
    t0 = time.time()
    clf.fit(X, Y)
    trainingTime = time.time() - t0
    
    
    t0 = time.time()
    print str(classifier[1])+" is now predicting"
    predictions = clf.predict(X_test)
    predictionTime = time.time() - t0
    accuracy = float(accuracy_score(Y_test, predictions, normalize=True))*100.0
    metricsValues =  precision_recall_fscore_support(Y_test, predictions, average='macro')
    precision = float(metricsValues[0])*100.0
    recall = float(metricsValues[1])*100.0
    
    print "the accuracy of this classifier is "+str(accuracy)
    print "the precision of this classifier is "+str(precision)
    print "the recall of this classifier is "+str(recall)
    print("train time: %0.3fs" % trainingTime)
    print("prediction time: %0.3fs" % predictionTime)
    
    print classification_report(Y_test, predictions)


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




featureSelectionList = ["linearSVM","SelectKBest","SelectKPercentile","TreeBased"]


print "=========================================================="
print "total number of feature selection processes being considered: "+str(len(featureSelectionList))
print "=========================================================="



classifierList = []
classifierList.append((linear_model.LogisticRegression(C=1e6,solver='liblinear'),"LogisticRegression"))
classifierList.append((RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"))
classifierList.append((Perceptron(n_iter=150), "Perceptron"))
classifierList.append((PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"))
classifierList.append((KNeighborsClassifier(n_neighbors=10), "kNN"))
classifierList.append((RandomForestClassifier(n_estimators=100), "Random forest"))
classifierList.append((AdaBoostClassifier(),"AdaBoost"))
classifierList.append((NearestCentroid(),"NearestCentroid"))
classifierList.append((SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),"SGD"))
classifierList.append((MultinomialNB(),"NaiveBayes"))
classifierList.append((ExtraTreesClassifier(),"ExtraTreesClassifier"))







X_training = X[0:200]
Y_training = Y[0:200]
X_test = X[200:437]
Y_test = Y[200:437]
for featureSelection in featureSelectionList:
    print "=========================================================="
    #featureSelectionProcess(X,Y,featureSelection)
    for classifier in classifierList:
        print "_____________________________"
        classifierPerformance(classifier,X_training,Y_training,X_test,Y_test)
        print "_______________________________"
    print "=========================================================="
    print "done"
    break
    
    

"""clf1 = linear_model.LogisticRegression(C=1e6,solver='liblinear')
clf2 = Perceptron(n_iter=150)
hybridClassifier(clf1,clf2,X_training,Y_training,X_test,Y_test)"""
    
    






# checking different classifier's performance





print "done"