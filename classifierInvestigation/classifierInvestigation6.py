#this code is to test the latency of feature extractions for vine and then instagram


import databaseCodes as dc
import numpy as np
from textblob import TextBlob
import time
import re
import urllib
import subprocess
import librosa 
import numpy as np
from os import listdir
from os.path import isfile, join

import warnings



warnings.filterwarnings("ignore")

#this is the location of the parent folder where all the codes and packages related to this work is stored.
parentPath = "C:\\Users\\RahatIbnRafiq\\workspace\\CyberSafetyClassifier\\"



class VineComment:
    commentText = ""
    userId = ""
    postId = ""
    commentType = ""
    polarity = 0.0 
    subjectivity = 0.0
    
    def __init__(self,commentText,userId,postId,commentType):
        self.commentText = commentText.lower()
        self.commentText = re.sub(r'[^\w\s]','',self.commentText )
        self.userId = userId 
        self.postId = postId 
        self.commentType = commentType   



class Vine:
    userDocumentList = []
    mediaSessionList = []
    mediaCommentDictionary = {}
    mediaURLMap = {}
    
    def __init__(self):
        self.userDocumentList = []
        #self.getVineUsersFromDatabase()
        #self.getVineMediasFromDatabase()
        #self.getVineCommentsFromDatabase()
        self.getVideoUrls()

    
    def getVideoUrls(self):
        print "getting media sessions urls from database"
        mediaDocuments = dc.findAllDataFromCollection("VineDatabase", "SampledASONAMPosts")
        for document in mediaDocuments:
            self.mediaURLMap[str(document["postId"])] = str(document["videoUrl"])
        print "getting media sessions urls from database is finished."
        
    #this function gets the vine users id from the mongo database and then writes them to a file
    def getVineUsersFromDatabase(self):
        print "getting users from database"
        userDocuments = dc.findAllDataFromCollection("VineDatabase", "SampledASONAMUsers")
        for document in userDocuments:
            self.userDocumentList.append(document)
        print "getting users from database is finished."
        self.userDocumentList = self.userDocumentList[0:10000]
    
    def getVineMediasFromDatabase(self):
        print "getting media sessions from database"
        mediaDocuments = dc.findAllDataFromCollection("VineDatabase", "SampledASONAMPosts")
        for document in mediaDocuments:
            self.mediaSessionList.append(document)
        print "getting media sessions from database is finished."
        self.mediaSessionList = self.mediaSessionList[0:983]
    
    def getVineCommentsFromDatabase(self):
        print "getting comments from database"
        commentDocuments = dc.findAllDataFromCollection("VineDatabase", "SampledASONAMComments")
        for document in commentDocuments:
            postId = str(document["postId"])
            userId = str(document["userId"])
            commentText = str(document["commentText"].encode("utf8"))
            if str(document["type"]) == "mention":
                commentType = "mention"
            elif str(document["type"]) == "tag":
                commentType = "tag"
            else:
                commentType = "other"
            comment = VineComment(commentText,userId,postId,commentType)
            if str(postId) in self.mediaCommentDictionary:
                self.mediaCommentDictionary[postId].append(comment)
            else:
                self.mediaCommentDictionary[postId] = []
                self.mediaCommentDictionary[postId].append(comment)
            
            
                
        print "getting comments from database is finished."
        
    
        


class SentimentExtraction:
    text = ""
    sentiment = None
    
    def __init__(self,text):
        self.text = text
        
    def getSentiment(self):
        blob = TextBlob(self.text)
        return blob.sentiment


class MediaFeatureExtraction:
    mediaInfoMatrix = None
    
    def __init__(self):
        self.mediaInfoMatrix = np.empty((983, 7))
    def extractMediaFeature(self,mediaDocumentList): 
        count = 0
        for media in mediaDocumentList:
            self.mediaInfoMatrix[count][0] = int(media["loopCount"])
            self.mediaInfoMatrix[count][1] = int(media["likeCount"])
            self.mediaInfoMatrix[count][2] = int(media["commentCount"])
            self.mediaInfoMatrix[count][3] = int(media["explicitContent"])
            self.mediaInfoMatrix[count][4] = int(media["repostCount"])
            description = str(media["description"].encode("utf8"))
            description =(description.decode('unicode_escape').encode('ascii','ignore'))
            sentiment = SentimentExtraction(str(description))
            descriptionSentiment = sentiment.getSentiment()
            self.mediaInfoMatrix[count][5] = float(descriptionSentiment.polarity)
            self.mediaInfoMatrix[count][6] = float(descriptionSentiment.subjectivity)
            count += 1




class UserFeatureExtraction:
    userInfoMatrix = None
    
    def __init__(self):
        self.userInfoMatrix = np.empty((10000, 10))
    def extractUserFeature(self,userDocumentList): 
        count = 0
        for user in userDocumentList:
            self.userInfoMatrix[count][0] = int(user["followerCount"])
            self.userInfoMatrix[count][1] = int(user["followingCount"])
            self.userInfoMatrix[count][2] = int(user["likeCount"])
            self.userInfoMatrix[count][3] = int(user["loopCount"])
            self.userInfoMatrix[count][4] = int(user["explicitContent"])
            self.userInfoMatrix[count][5] = int(user["repostsEnabled"])
            self.userInfoMatrix[count][6] = int(user["postCount"])
            self.userInfoMatrix[count][7] = int(user["authoredPostCount"])
            description = str(user["description"].encode("utf8"))
            sentiment = SentimentExtraction(str(description))
            descriptionSentiment = sentiment.getSentiment()
            self.userInfoMatrix[count][8] = float(descriptionSentiment.polarity)
            self.userInfoMatrix[count][9] = float(descriptionSentiment.subjectivity)
            count += 1
                   


class CommentFeatureExtraction():
    negativeWordMap = {}
    
    def loadNegativeWords(self):
        f = open("new_neg_list1.txt","r")
        for line in f:
            self.negativeWordMap[line.strip()] = 1
        f.close()
    
    def __init__(self):
        self.loadNegativeWords()
    
    def extractCommentFeature(self,mediaCommentDictionary):
        count = 0
        for mediaid in mediaCommentDictionary:
            tagCount = 0
            mentionCount = 0
            otherCount = 0
            negativeCommentCount = 0
            totalCommentCount = float(len(mediaCommentDictionary[mediaid]))
            totalNegativeCommentPolarity = 0.0 
            totalNegativeCommentSubjectivity = 0.0
            allCommentPolarity = 0.0 
            allCommentSubjectivity = 0.0
            negativeWordCount = 0.0
            for comment in mediaCommentDictionary[mediaid]:
                if comment.commentType == "mention":
                    mentionCount+=1
                elif comment.commentType == "tag":
                    tagCount+=1
                else:
                    otherCount+=1
                try:
                    sentiment = SentimentExtraction(comment.commentText)
                    comment.polarity = float(sentiment.getSentiment().polarity)
                    comment.subjectivity = float(sentiment.getSentiment().subjectivity)
                    
                    allCommentPolarity = allCommentPolarity + comment.polarity
                    allCommentSubjectivity = allCommentSubjectivity + comment.subjectivity
                    
                    if comment.polarity <0.0:
                        negativeCommentCount+=1
                        totalNegativeCommentPolarity = totalNegativeCommentPolarity + comment.polarity
                        totalNegativeCommentSubjectivity = totalNegativeCommentSubjectivity + comment.subjectivity
                        
                        words = comment.commentText.split(" ")
                        for word in words:
                            if word in self.negativeWordMap:
                                negativeWordCount+=1
                        
                except Exception:
                    continue
            count += 1
            if count == 100:
                break
            



class audioFeatureExtraction:
    def __init__(self):
        print "hello"
    def downloadVideos(self,mediaURLMap):
        for mediaid in mediaURLMap:
            urllib.urlretrieve (mediaURLMap[mediaid],"videoFolder\\"+mediaid+".mp4")
    
    def convertVideosToAudios(self,mediaURLMap):
        
        for mediaid in mediaURLMap:
            str1 =  "ffmpeg -i C:/Users/RahatIbnRafiq/Desktop/videoFolder/"+mediaid+".mp4"
            str2 = " -ab 160k -ac 2 -ar 44100 -vn C:/Users/RahatIbnRafiq/Desktop/audioFolder/"+mediaid+".wav"
            command = str1+str2 
            subprocess.call(command, shell=True)
            
    def audioFeatureExtraction(self):
        files = [ f for f in listdir("C:/Users/RahatIbnRafiq/Desktop/audioFolder/") if isfile(join("C:/Users/RahatIbnRafiq/Desktop/audioFolder/",f)) ]
        for filename in files:
            audio_path = "C:/Users/RahatIbnRafiq/Desktop/audioFolder/"+filename
            y, sr = librosa.load(audio_path)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_S = librosa.logamplitude(S, ref_power=np.max)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
            S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)
            log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
            log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)
            C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            mfcc= librosa.feature.mfcc(S=log_S, n_mfcc=13)
            delta_mfcc  = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
            print 'Estimated tempo:        %.2f BPM' % tempo
            print 'First 5 beat frames:   ', beats[:5]
            print 'First 5 beat times:    ', librosa.frames_to_time(beats[:5], sr=sr)
            C_sync = librosa.feature.sync(C, beats, aggregate=np.median)
            print C_sync.shape
            

vine = Vine()

afe = audioFeatureExtraction()
print str(len(vine.mediaURLMap))+" total media extracted."


startTime =  time.time() * 1000
afe.convertVideosToAudios(vine.mediaURLMap)
endTime =  time.time() * 1000
conversionTime = endTime-startTime



startTime =  time.time() * 1000
afe.audioFeatureExtraction()
endTime =  time.time() * 1000
extractionTime = endTime - startTime


print str(extractionTime)+" time for feature extraction"
print str(conversionTime)+" time for video to audio conversion"





print "done"