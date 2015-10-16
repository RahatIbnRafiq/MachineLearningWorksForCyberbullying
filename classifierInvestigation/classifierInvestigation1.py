
import random 
from pymongo import MongoClient
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
import time

from textblob import TextBlob
import nltk
from nltk import bigrams
from nltk import trigrams
from lasagne.layers.noise import dropout
import math
import random




def FeatureExtractionVersionTwo(data):
    featureDictionary = defaultdict(int)
    
    #user features
    featureDictionary["userFollowerCount"] = int(data["userFollowerCount"])
    featureDictionary["userFollowingCount"] = int(data["userFollowingCount"])
    featureDictionary["userLikeCount"] = int(data["userLikeCount"])
    featureDictionary["userloopCount"] = int(data["userloopCount"])
    featureDictionary["userPostCount"] = int(data["userPostCount"])
           

    return featureDictionary
    

def SplitIntoTestTrainingDataset(filename):
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
                    featureDictionary = FeatureExtractionVersionTwo(ii)
                    testData.append(featureDictionary)
                    testLabel.append(label)
                else:
                    featureDictionary = FeatureExtractionVersionTwo(ii)
                    trainingData.append(featureDictionary)
                    trainingLabel.append(label)

            else:
                label = 1
                bullyingCount = bullyingCount + 1
                if bullyingCount > 179:   #179
                    continue
                if bullyingCount > 100:  #100
                    featureDictionary = FeatureExtractionVersionTwo(ii)
                    testData.append(featureDictionary)
                    testLabel.append(label)
                else:
                    featureDictionary = FeatureExtractionVersionTwo(ii)
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
    




def trainingLogistocRegression():
    
    print "=========================================================="
    totalData,totalLabel,index = SplitIntoTestTrainingDataset("vine_meta_data.csv")
    print "getting data is finished successfully"
    print "index of test and training cut off"+str(index)
    print "=========================================================="
    print "=========================================================="
    print "now transforming data into vectorized form"
    X,Y = TransformIntoVectors(totalData,totalLabel)
    print "data has been convereted into vectorized form successfully"
    print "=========================================================="
    X_Matrix,Y_Matrix = TransformIntoMatrix(X,Y)
    trainingSet  =  X_Matrix[0:437,0:5]
    logReg = linear_model.LogisticRegression(C=1e6)
    logReg.fit(trainingSet, Y_Matrix)
    return logReg
    

def findFromDatabaseUsers(mediaid):
    client = MongoClient()
    db = client.classifierInvestigation
    cursor = db.users.find({"_id": str(mediaid)})[0]
    return cursor

def insertIntoDatabaseUsers(mediaid,followerCount,followingCount,likeCount,loopCount,postCount,logisticValue):
    client = MongoClient()
    db = client.classifierInvestigation
    try:
        db.users.insert_one(
        {
            "_id": str(mediaid),
            "followerCount": str(followerCount),
            "followingCount":str(followingCount),
            "likeCount": str(likeCount),
            "loopCount": str(loopCount),
            "postCount": str(postCount),
            "logisticValue" : str(logisticValue)
        })
        return "success"
    except Exception as e:
        if "duplicate key error" in str(e):
            return "failure"



def manualPredict(theta, X,y,intercept):
    h = X.dot(theta.T)+intercept
    m = h.size-1
    accuracy = 0.0
    while m > -1:
        if h[m] > 0.0:
            predict = 1.0
        else:
            predict = 0.0
        if predict == y[m]:
            accuracy = accuracy+1.0
        m = m - 1
    return predict





#checking non incremental version

mediaIdLimits = [100,1000,10000]
followerCountList = [50,500,5000]
followingCountList = [50,500,5000]
likeCountList = [50,500,5000]
loopCountList = [50,500,5000]
postCountList = [50,500,5000]
logReg = trainingLogistocRegression()
f = open("classiferNotIncremental.txt","w")
for mediaIdLimit in mediaIdLimits:
    startTime =  time.time() * 1000
    for mediaid in range(1,mediaIdLimit+1):
        followerCount = random.choice(followerCountList)
        followingCount = random.choice(followingCountList)
        likeCount = random.choice(likeCountList)
        loopCount = random.choice(loopCountList)
        postCount = random.choice(postCountList)
        randomTestMatrix = np.empty((1, 5))
        randomTestMatrix[0][0] = followerCount
        randomTestMatrix[0][1] = followingCount
        randomTestMatrix[0][2] = likeCount
        randomTestMatrix[0][3] = loopCount
        randomTestMatrix[0][4] = postCount
        randomTestMatrix = np.reshape(randomTestMatrix[0], (1, randomTestMatrix[0].shape[0]))
        prediction = logReg.predict(randomTestMatrix)[0]
    endTime =  time.time() * 1000
    print "total time: "+str(endTime - startTime)
    print "number of instances: "+str(mediaIdLimit)
    f.write(str(mediaIdLimit)+","+str(endTime - startTime)+"\n")
    print "_____________________________________"  
    

f.close()





# check incremental version
"""
mediaIdLimits = [10000]
followerCountList = [50,500,5000]
followingCountList = [50,500,5000]
likeCountList = [50,500,5000]
loopCountList = [50,500,5000]
postCountList = [50,500,5000]
logReg = trainingLogistocRegression()

theta = logReg.coef_[0]
intercept =  logReg.intercept_[0]


for mediaIdLimit in mediaIdLimits:
    startTime =  time.time() * 1000
    for mediaid in range(1,mediaIdLimit+1):
        followerCount = random.choice(followerCountList)
        followingCount = random.choice(followingCountList)
        likeCount = random.choice(likeCountList)
        loopCount = random.choice(loopCountList)
        postCount = random.choice(postCountList)
        logisticValue = theta[0]*followerCount + theta[1]*followingCount + theta[2]*likeCount + theta[3]*loopCount + theta[4]*postCount
        insertIntoDatabaseUsers(mediaid, followerCount, followingCount, likeCount, loopCount, postCount,logisticValue)
    break

f = open("classifierIncremental.txt","a")
for mediaIdLimit in mediaIdLimits:
    startTime =  time.time() * 1000
    databaseTime = 0.0
    for mediaid in range(1,mediaIdLimit+1):
        followerCount = random.choice(followerCountList)
        followingCount = random.choice(followingCountList)
        likeCount = random.choice(likeCountList)
        loopCount = random.choice(loopCountList)
        postCount = random.choice(postCountList)
        databaseTimeStart = time.time() * 1000
        userData = findFromDatabaseUsers(mediaid)
        if followerCount != userData["followerCount"]:
            userData["logisticValue"] = str(float(userData["logisticValue"]) - theta[0]*float(userData["followerCount"]) + theta[0]*followerCount)
        databaseTimeEnd = time.time() * 1000
        databaseTime = databaseTime + float(databaseTimeEnd-databaseTimeStart)
        finalValue = float(userData["logisticValue"]) + intercept
        if finalValue > 0.0:
            prediction = 1 
        else:
            prediction = 0
    endTime =  time.time() * 1000
    print "total time: "+str(endTime - startTime)
    print "database related time: "+str(databaseTime)
    print "number of instances: "+str(mediaIdLimit)
    f.write(str(mediaIdLimit)+","+str(endTime - startTime)+","+str(databaseTime)+"\n")
    print "_____________________________________"  
    break
f.close()

"""

print "done"