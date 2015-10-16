'''
Created on Feb 5, 2015

@author: RahatIbnRafiq
'''



from __builtin__ import str


import json


class VinePost:
    def __init__(self,videoDashUrl,foursquareVenueId,userId,private,loopCount,thumbnailUrl,
                 explicitContent,avatarUrl,commentCount,videoLowURL,username,description,
                 permalinkUrl,postId,likeCount,created,videoUrl,repostCount,shareUrl):
        self.videoDashUrl = videoDashUrl
        self.foursquareVenueId = foursquareVenueId
        self.userId = userId
        self.private = userId
        self.loopCount = loopCount
        self.thumbnailUrl = thumbnailUrl
        self.explicitContent = explicitContent
        self.avatarUrl = avatarUrl
        self.commentCount = commentCount
        self.videoLowURL = videoLowURL
        self.username = username
        self.description = description
        self.permalinkUrl = permalinkUrl
        self.postId = postId
        self.likeCount = likeCount
        self.created = created
        self.videoUrl = videoUrl
        self.repostCount = repostCount
        self.shareUrl = shareUrl

def parsePostData(post):
    videoDashUrl= str(post['videoDashUrl'])
    foursquareVenueId= str(post['foursquareVenueId'])
    userId= str(post['userId'])
    private= str(post['private'])
    loopCount= str(post['loops']['count'])
    thumbnailUrl = str(post['thumbnailUrl'])
    explicitContent = str(post['explicitContent'])
    avatarUrl = str(post['avatarUrl'])
    commentCount = str(post['comments']['count'])
    videoLowURL= str(post['videoLowURL'])
    username= str(post['username'].encode('utf-8'))
    description= str(post['description'].encode('utf-8'))
    permalinkUrl= str(post['permalinkUrl'])
    postId= str(post['postId'])
    likeCount = str(post['likes']['count'])
    created= str(post['created'])
    videoUrl= str(post['videoUrl'])
    repostCount = str(post['reposts']['count'])
    shareUrl = str(post['shareUrl'])
    newpost = VinePost(videoDashUrl,foursquareVenueId,userId,private,loopCount,thumbnailUrl,explicitContent,avatarUrl,commentCount,videoLowURL,username,description,permalinkUrl,postId,likeCount,created,videoUrl,repostCount,shareUrl)
    return newpost

       
        

def getAllPosts(vineList,userid,apiCount):
    page = 1
    vine = vineList[apiCount]
    postList = []
    
    while True:
        try:
            posts = vine.get_user_timeline(user_id=userid,size='100',page = page)
            if len(posts) <= 0:
                    break
            for post in posts:
                metadata = post['json']
                data = json.loads(str(metadata))
                newPost = parsePostData(data)
                postList.append(newPost)
            page = page + 1
            print "page:"+str(page)
        except Exception as e:
            if "permission" in str(e):
                print "private user."
                return None
            if "try again later" in str(e): 
                apiCount = apiCount + 1
                apiCount = apiCount % 7
                vine = vineList[apiCount]
                print "Switching API. now API is :"+str(apiCount)
                posts = vine.get_user_timeline(user_id=userid,size='100',page = page)
                if len(posts) <= 0:
                    break
                for post in posts:
                    metadata = post['json']
                    data = json.loads(str(metadata))
                    newPost = parsePostData(data)
                    postList.append(newPost)
                page = page + 1
                print "page:"+str(page)
            else:
                print str(e)
                print "Final Exception!"
                return None

            
    return (postList,apiCount)


def getSinglePost(vineList,postid,apiCount):
    vine = vineList[apiCount]
    post = None
    try:
        post = vine.get_post(post_id=postid)
        metadata = post['json']
        data = json.loads(str(metadata))
        newPost = parsePostData(data["records"][0])
        return (newPost,apiCount)
    except Exception as e:
        if "permission" in str(e):
            print "private post."
            return (None,apiCount)
        if "try again later" in str(e): 
            apiCount = apiCount + 1
            apiCount = apiCount % 4
            vine = vineList[apiCount]
            print "Switching API. now API is :"+str(apiCount)
            try:
                post = vine.get_post(post_id=postid)
                metadata = post['json']
                data = json.loads(str(metadata))
                newPost = parsePostData(data["records"][0])
                return (newPost,apiCount)
            except Exception:
                return (None,apiCount)
        else:
            print str(e)
            print "Final Exception in getSinglePost!"
            print str(e)
            return (None,apiCount)
    return (newPost,apiCount)


