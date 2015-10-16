

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from csv import DictReader
from collections import defaultdict
from numpy import array





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
    




def generateTrainingDataset():
    totalData,totalLabel,index = SplitIntoTestTrainingDataset("vine_meta_data.csv")
    X,Y = TransformIntoVectors(totalData,totalLabel)
    X_Matrix,Y_Matrix = TransformIntoMatrix(X,Y)
    trainingSet  =  X_Matrix[0:437,0:5]
    return (trainingSet, Y_Matrix)


