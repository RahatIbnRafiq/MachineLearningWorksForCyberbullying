'''
Created on Jan 30, 2015

@author: RahatIbnRafiq
'''


class FollowerUser:
    def __init__(self,userid,username,verified,followId,Uid,location):
        self.userid = userid
        self.username = username
        self.verified = verified
        self.followId = followId
        self.Uid = Uid
        self.location = location

    
        

def getAllFollowers(vineList,userid,apiCount):
    usersList = []
    page = 1
    followerUserList = []
    users = []
    vine = vineList[apiCount]
    
    while True:
        try:
            users = vine.get_followers(user_id=userid,size='200',page=str(page))
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
                print "private user"
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
            newUser = FollowerUser(userid,username,verified,followId,Uid,location)
            followerUserList.append(newUser) 
            
    return (followerUserList,apiCount)
