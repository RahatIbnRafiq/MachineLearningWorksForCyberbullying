

"""
liked
videoDashUrl
foursquareVenueId
userId
private
api
videoWebmUrl
loops
thumbnailUrl
explicitContent
myRepostId
blocked
verified
avatarUrl
description
id
entities
json
videoLowURL
_attrs
username
vanityUrls
tags
permalinkUrl
promoted
user
postId
videoUrl
name
followRequested
created
shareUrl
profileBackground
following
"""
import getVineAPI
import warnings
import json
import getPost
import numpy as np
import mlpy
warnings.filterwarnings("ignore")




def getPostIds(filename):
    f = open(filename,"r")
    g = open("fortyPercentOrMorePosts.txt","a")
    count = 0
    for line in f:
        line = line.strip()
        values = line.split(",")
        try:
            negativity =float( values[3])
            negativity = int(negativity)
            if negativity > 39:
                count = count + 1
                g.write(str(line)+"\n")
        except Exception:
            continue
    g.close()

def writePostIdsToFile():

    filenames = ["postids_mentioncount_negativity_positivity_gt15lt30.txt","postids_mentioncount_negativity_positivity_gt170lt400.txt",
                 "postids_mentioncount_negativity_positivity_gt30lt50.txt","postids_mentioncount_negativity_positivity_gt50lt90.txt",
                 "postids_mentioncount_negativity_positivity_gt90lt170.txt"]
    for filename in filenames[0:5]:
        getPostIds(filename)

def writePostInformationToTheFile(filename):
    count = 0
    vineList = getVineAPI.getVineAPI()
    apiCount = 1
    vine = vineList[apiCount]
    f = open(filename,"r")
    
    for line in f:
        line = line.strip()
        values = line.split(",")
        postid = values[0]
        try:
            post = vine.get_post(post_id=postid)
            count = count + 1
            print count
            metadata = post['json']
            data = json.loads(str(metadata))
            newPost = getPost.parsePostData(data["records"][0])
            negativity = float(values[3])
            if negativity > 60.0:
                g = open("trainingData.txt","a")
                g.write(str(postid)+","+str(newPost.likeCount)+","+str(newPost.commentCount)+","+str(newPost.repostCount)+","+str("1")+"\n")
                g.close()
            else:
                g = open("trainingData.txt","a")
                g.write(str(postid)+","+str(newPost.likeCount)+","+str(newPost.commentCount)+","+str(newPost.repostCount)+","+str("0")+"\n")
                g.close()
        except Exception:
            continue
    g.close()
    f.close()
#writePostIdsToFile()


#writePostInformationToTheFile("fortyPercentOrMorePosts.txt")



x, y = mlpy.data_fromfile('data.dat') 




print "done"
