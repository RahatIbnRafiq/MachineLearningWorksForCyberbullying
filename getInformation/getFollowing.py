'''
Created on Jan 20, 2015

@author: RahatIbnRafiq
'''
from __builtin__ import str

"""
username:Zachary Piona
verified:0
vanityUrls:[]
followRequested:0


avatarUrl:http://v.cdn.vine.co/r/avatars/F3A55C5F671108334494769135616_2b5529fa32a.1.2.jpg?versionId=756OzGWvs0L_EdA4kRbzacKUydjIcwVB
api:<vinepy.api.API object at 0x02901030>
userId:920618402438057984
location:
profileBackground:0x68bf60
name:Zachary Piona
followId:1164469210756562944
json:{"username": "Zachary Piona", "verified": 0, "vanityUrls": [], "followRequested": 0, "avatarUrl": "http://v.cdn.vine.co/r/avatars/F3A55C5F671108334494769135616_2b5529fa32a.1.2.jpg?versionId=756OzGWvs0L_EdA4kRbzacKUydjIcwVB", "userId": 920618402438057984, "profileBackground": "0x68bf60", "followId": 1164469210756562944, "notifyPosts": 0, "location": "", "following": 0, "blocked": 0, "user": {"private": 0}}
notifyPosts:0
'User' object has no attribute 'encode'
blocked:0
id:920618402438057984
"""



class FollowingUser:
    def __init__(self,userid,username,verified,followId,Uid,location):
        self.userid = userid
        self.username = username
        self.verified = verified
        self.followId = followId
        self.Uid = Uid
        self.location = location

    
        

def getAllFollowings(vineList,userid,apiCount):
    usersList = []
    page = 1
    followingUserList = []
    users = []
    vine = vineList[apiCount]
    
    while True:
        try:
            users = vine.get_following(user_id=userid,size='200',page=str(page))
            if len(users) <= 0:
                    break
            usersList.append(users)
            page = page + 1
            print "page:"+str(page)
        except Exception as e:
            if "permission" in str(e):
                print "private user."
                return ("private",apiCount)
            if "try again later" in str(e): 
                apiCount = apiCount + 1
                apiCount = apiCount % 4
                vine = vineList[apiCount]
                print "swicthing api. now api is :"+str(apiCount)
                users = vine.get_following(user_id=userid,size='200',page=page)
                if len(users) <= 0:
                    break
                usersList.append(users)
                page = page + 1
                print "page:"+str(page)
            else:
                print str(e)
                print "hahahaha"
                return ("private",apiCount)
    
    for users in usersList:
        for user in users:
            userid = str(user['userId'])
            username = str(user['username'].encode('ascii', 'ignore').decode('ascii'))
            verified = str(user['verified'])
            followId = str(user['followId'])
            if user['location'] is not None :
                location = str(user['location'].encode('ascii', 'ignore').decode('ascii'))
            else:
                location = ""
            Uid = str(user['id'])
            newUser = FollowingUser(userid,username,verified,followId,Uid,location)
            followingUserList.append(newUser) 
            
    return (followingUserList,apiCount)
