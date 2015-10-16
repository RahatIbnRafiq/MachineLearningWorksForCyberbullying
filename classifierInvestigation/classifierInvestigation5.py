#this code is to test the latency of feature extractions for vine and then instagram

import generateTraining as gt
import databaseCodes as dc
import vinepy
import getInformation as getinfo
import warnings


from sklearn import linear_model
import numpy as np
import time

warnings.filterwarnings("ignore")

#this is the location of the parent folder where all the codes and packages related to this work is stored.
parentPath = "C:\\Users\\RahatIbnRafiq\\workspace\\CyberSafetyClassifier\\"





class TestData:
    test_x = arrayOfData = np.zeros((100,6),dtype=np.float)


# this class instantiates one object with the key's username, password and sessionkey
class VineAPIInstance:
    username = ""
    password = ""
    sessionkey = ""
    api = None
    def __init__(self,username,password,sessionkey):
        self.username = username
        self.password = password
        self.sessionkey = sessionkey
    def createAPI(self):
        api = vinepy.API(username=self.username, password=self.password,session_id=self.sessionkey)
        return api




class VineAPIClass:
    vineAPIList = None
    def __init__(self):
        self.vineAPIList = []
    
    #this function gets the api sessionkeys from the file textFiles/keys1.txt
    def getAPIKeys(self):
        f = open(parentPath+"textFiles//keys1.txt","r")
        for line in f:
            line =  line.strip()
            data = line.split(",")
            username = str(data[0])
            password = str(data[1])
            sessionkey = str(data[2])
            api = VineAPIInstance(username,password,sessionkey)
            self.vineAPIList.append(api.createAPI())
        f.close()
    



class Vine:
    userIDList = None
    vineAPIList = None
    vineUserList = []
    test = None
    
    def __init__(self):
        self.userIDList = []
        self.vineAPIList = []
        self.getVineUsersFromDatabase()
        self.getVineAPIList()

        
    def getVineAPIList(self):
        print "getting session keys to collect data"
        vine = VineAPIClass()
        vine.getAPIKeys()
        self.vineAPIList = vine.vineAPIList
        print "session key collection is done"
        print "total keys collected: "+str(len(self.vineAPIList))
        print "##########################################"
    
    #this function gets the vine users id from the mongo database and then writes them to a file
    def getVineUsersFromDatabase(self):
        print "getting userids from database"
        userDocuments = dc.findAllDataFromCollection("VineDatabase", "ASONAMCollectedUser")
        for document in userDocuments:
            self.userIDList.append(document['userId'])
            
        self.userIDList = self.userIDList[0:1000]
        print "getting userids from database is done."
        print "total users collected: "+str(len(self.userIDList))
        print "##########################################"
        
    #this function returns the number of users collected from the VineDatabase collection in mongodb
    def getNumberOfUsersCollectedFromDatabase(self):
            return len(self.userIDList)
        
    def collectUserInformationFromAPI(self):
        self.test =np.zeros((len(self.userIDList),6),dtype=np.float)
        apiCount = 0
        count = 0
        for userid in self.userIDList:
            returnData = None
            try:
                returnData = getinfo.getBio.getBioOfUser(self.vineAPIList, userid, apiCount,3)
            except Exception as ex1:
                print ex1
                continue
            if returnData is None:
                print "return data is none"
                continue
            elif returnData[0] is None:
                print "return data[0] is none"
                continue
            else:
                user = returnData[0]
                user.usernumber = count
                self.vineUserList.append(user)
                
                self.test[user.usernumber][0] = int(user.followerCount)
                self.test[user.usernumber][1] = int(user.followingCount)
                self.test[user.usernumber][2] = int(user.likeCount)
                self.test[user.usernumber][3] = int(user.loopCount)
                self.test[user.usernumber][4] = int(user.postCount)
                self.test[user.usernumber][5] = 0
                apiCount = returnData[1]
                count = count + 1
                

class TrainingData:
    training_x = None
    training_y = None
    def __init__(self):
        self.training_x, self.training_y = gt.generateTrainingDataset() 

class LogisticRegressionApply():
    logReg = None
    theta = None
    intercept = None
    def __init__(self):
        self.logReg = linear_model.LogisticRegression(C=1e6)
        startTrainingTime = time.time() * 1000
        tr = TrainingData()
        self.logReg.fit(tr.training_x, tr.training_y)
        endTrainingTime = time.time() * 1000
        
        timeSpentTraining = endTrainingTime - startTrainingTime
        print "total time training: "+str(timeSpentTraining)
        
        #self.theta = self.logReg.coef_[0]
        #self.intercept =  self.logReg.intercept_[0]
    
    def processTestRow(self,testRow):
        randomTestMatrix = np.empty((1, 5))
        randomTestMatrix[0][0] = testRow[0]
        randomTestMatrix[0][1] = testRow[1]
        randomTestMatrix[0][2] = testRow[2]
        randomTestMatrix[0][3] = testRow[3]
        randomTestMatrix[0][4] = testRow[4]
        randomTestMatrix = np.reshape(randomTestMatrix[0], (1, randomTestMatrix[0].shape[0]))
        return randomTestMatrix
        
    def nonIncremental(self,vine):
        print "starting non-incremental version"
        
        startFeatureExtraction = time.time() * 1000
        vine.collectUserInformationFromAPI()
        endFeatureExtraction = time.time() * 1000
        
        timeSpentFeatureExtraction = endFeatureExtraction - startFeatureExtraction
        
        
        startTimePrediction =  time.time() * 1000
        for testRow in vine.test:
            randomTestMatrix = self.processTestRow(testRow)
            self.logReg.predict(randomTestMatrix)[0]
        endTimePrediction =  time.time() * 1000
        
        timeSpentPrediction = endTimePrediction - startTimePrediction
        print "total time feature extraction: "+str(timeSpentFeatureExtraction)
        print "total time predicting: "+str(timeSpentPrediction)
        return (timeSpentFeatureExtraction,timeSpentPrediction)
            
            
            


vine = Vine()

logReg = LogisticRegressionApply()
totalTimeFeatureExtraction = 0.0
totalTimePrediction = 0.0

numberOfRounds = 3

for x in range(1,numberOfRounds+1):
    print "round: "+str(x)
    p,q = logReg.nonIncremental(vine)
    totalTimeFeatureExtraction = totalTimeFeatureExtraction + p
    totalTimePrediction = totalTimePrediction + q
    print "__________________"
    
print "average time spent feature extraction: "+str(totalTimeFeatureExtraction/numberOfRounds)
print "average time spent prediction: "+str(totalTimePrediction/numberOfRounds)


print "done"