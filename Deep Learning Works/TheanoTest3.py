
from sklearn.feature_extraction import DictVectorizer
from csv import DictReader
from collections import defaultdict
from numpy import array
import numpy as np

import nltk
from nltk import bigrams
from nltk import trigrams

import theano
from theano import tensor as T
from sklearn.cross_validation import train_test_split



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
    
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    
    return train_test_split(all_X, all_Y, test_size=0.33)
                
    

def init_weights(shape):
    """ Weight initialization """
    weights = np.asarray(np.random.randn(*shape) * 0.01, dtype=float)
    return theano.shared(weights)

def backprop(cost, params, lr=0.5):
    """ Back-propagation """
    grads   = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def forwardprop(X, w_1, w_2):
    """ Forward-propagation """
    h    = T.nnet.sigmoid(T.dot(X, w_1))  # The \sigma function
    yhat = T.nnet.softmax(T.dot(h, w_2))  # The \varphi function
    return yhat   
        
    
    

emotionDictionary = loadEmotionDictionary()
contentdictionary =  loadContentDictionary()

populateNegWord()
print "=========================================================="
totalData,totalLabel,index = SplitIntoTestTrainingDataset("vine_meta_data.csv",emotionDictionary,contentdictionary)
print "getting data is finished successfully"
print "=========================================================="



print "=========================================================="
print "now transforming data into vectorized form"
data,target = TransformIntoVectors(totalData,totalLabel)
print "data has been convereted into vectorized form successfully"
print "=========================================================="



print "=========================================================="
print "now transforming vector into matrix form"
train_X, test_X, train_y, test_y = TransformIntoMatrix(data,target)
print "data has been convereted into vectorized form successfully"
print "=========================================================="





X = T.fmatrix()
Y = T.fmatrix()


# Layer's sizes
x_size = train_X.shape[1]             # Number of input nodes: 4 features and 1 bias
h_size = 500                          # Number of hidden nodes
y_size = train_y.shape[1]             # Number of outcomes (3 iris flowers)
w_1 = init_weights((x_size, h_size))  # Weight initializations
w_2 = init_weights((h_size, y_size))


# Forward propagation
yhat   = forwardprop(X, w_1, w_2)

# Backward propagation
cost    = T.mean(T.nnet.categorical_crossentropy(yhat, Y))
params  = [w_1, w_2]
updates = backprop(cost, params)



# Train and predict
train   = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
pred_y  = T.argmin(yhat, axis=1)
predict = theano.function(inputs=[X], outputs=pred_y, allow_input_downcast=True)



# Run SGD
"""for iter in range(500):
    train(train_X, train_y)
    train_accuracy = np.mean(np.argmax(train_y, axis=1) == predict(train_X))
    test_accuracy  = np.mean(np.argmax(test_y, axis=1) == predict(test_X))
    print predict(test_X)
    print("Iteration = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
            % (iter + 1, 100 * train_accuracy, 100 * test_accuracy))
    break"""
          
train(train_X, train_y)
print predict(test_X)
print "done"

