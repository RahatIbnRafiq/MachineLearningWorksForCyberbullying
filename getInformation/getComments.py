'''
Created on Feb 5, 2015

@author: RahatIbnRafiq
'''

"""

json:{"comment": "Fannyemiliaberglund", "username": "\u00af\\_(\u30c4)_/\u00af", 
"verified": 0, "vanityUrls": [], "avatarUrl": "http://v.cdn.vine.co/r/avatars/CF8DDF6D701117412969996079104_2755fd865b7.5.1.jpg?versionId=6D
MmnnR.UaoyEU93bKO5RBK.MC6wo.zs", "flags|platform_lo": 0, "userId": 953271863428132864, "created": "2014-12-12T10:59:44.000000", 
"entities": [{"vanityUrls": [], "title": "Fannyemiliaberglund", "range": [0, 19], "link": "vine://user-id/1035344122657357824", 
"type": "mention", "id": 1035344122657357824}], "location": "in mr. griers bed", "commentId": 1155143912516743168, 
"postId": 1154888354702503936, "flags|platform_hi": 0, "user": {"username": "\u00af\\_(\u30c4)_/\u00af", "verified": 0, "description": 
"\ud83c\udf54\ud83c\udf54\ud83c\udf5f\ud83c\udf5f = My Life", "avatarUrl": "http://v.cdn.vine.co/r/avatars/CF8DDF6D701117412969996079104_
2755fd865b7.5.1.jpg?versionId=6DMmnnR.UaoyEU93bKO5RBK.MC6wo.zs", "userId": 953271863428132864, "location": "in mr. griers bed", 
"profileBackground": "0xf279ac", "private": 0, "vanityUrls": []}}
"""


from __builtin__ import str


import json


class VineComment:
    def __init__(self,commentId,commentText,username,verified,avatarUrl,userId,created,title,link,commentType,location,description,private,postId):
        
        self.commentId  = commentId
        self.postId  = postId
        self.commentText = commentText
        self.username = username
        self.verified = verified
        self.avatarUrl = avatarUrl
        self.userId = userId
        self.created = created
        self.title = title
        self.link = link
        self.commentType = commentType
        self.location = location
        self.description = description
        self.private = private

def parseCommentData(comment):
    
    commentId = comment['commentId']
    postId = comment['postId']
    commentText = str(comment['comment'].encode('utf-8'))
    username= str(comment['username'].encode('utf-8'))
    verified= comment['verified']
    avatarUrl= comment['avatarUrl']
    userId= comment['userId']
    created= comment['created']
    try:
        title= comment['entities'][0]['title']
    except Exception:
        title= ""
    try:
        commentType= comment['entities'][0]['type']
    except Exception:
        commentType= ""
    try:
        link= comment['entities'][0]['link']
    except Exception:
        link= ""
    try:
        location= str(comment['location'].encode('utf-8'))
    except Exception:
        location= ""
    
    try:
        description= str(comment['user']['description'].encode('utf-8'))
    except Exception:
        description= ""
    
    private= comment['user']['private']

    newComment = VineComment(commentId,commentText,username,verified,avatarUrl,userId,created,
                             title,link,commentType,location,description,private,postId)
    return newComment




def getAllComments(vineList,postid,apiCount):
    page = 1
    vine = vineList[apiCount]
    commentList = []
    
    while True:
        try:
            comments = vine.get_post_comments(post_id=postid,size='100',page=page)
            if len(comments) <= 0:
                    break
            for comment in comments:
                metadata = comment['json']
                data = json.loads(str(metadata))
                newComment = parseCommentData(data)
                commentList.append(newComment)
            page = page + 1
            print "page:"+str(page)
        except Exception as e:
            if "permission" in str(e):
                print "private user."
                return None
            if "try again later" in str(e): 
                apiCount = apiCount + 1
                apiCount = apiCount % 4
                vine = vineList[apiCount]
                print "Switching API. now API is :"+str(apiCount)
                try:
                    comments = vine.get_post_comments(post_id=postid,size='100',page=page)
                except Exception as ex:
                    break
                if len(comments) <= 0:
                    break
                for comment in comments:
                    metadata = comment['json']
                    data = json.loads(str(metadata))
                    newComment = parseCommentData(data)
                    commentList.append(newComment)
                page = page + 1
                print "page:"+str(page)
            else:
                print str(e)
                print "Final Exception!"
                return None

            
    return (commentList,apiCount)

