'''
Created on Jan 20, 2015

@author: RahatIbnRafiq
'''
"""
followerCount:1563586
userId:953407442820173824
private:0
likeCount:1338
postCount:96
api:<vinepy.api.API object at 0x029D10B0>
explicitContent:0
id:953407442820173824
description:The beatboxer guy                                                    IG/Twitter/Snapchat  Markaaaay                       LA county
verified:0
loopCount:407046259
avatarUrl:http://v.cdn.vine.co/r/avatars/32750AA2911020482848441946112_pic-r-1386276280332a6588a537d.jpg_jV4ONJF4pmLaHJlxy6Cy5dRT60HXPnnIm718WpzgrSBx4yO4ePWrqFd3bsS2pePf.jpg?versionId=_biM486XW_7f6m2KWyi8VpfjrERh9EMG
authoredPostCount:86
json:{"followerCount": 1563586, "userId": 953407442820173824, "private": 0, "likeCount": 1338, "postCount": 96, "explicitContent": 0, "vanityUrls": ["MarcusPerez"], "verified": 0, "loopCount": 407046259, "avatarUrl": "http://v.cdn.vine.co/r/avatars/32750AA2911020482848441946112_pic-r-1386276280332a6588a537d.jpg_jV4ONJF4pmLaHJlxy6Cy5dRT60HXPnnIm718WpzgrSBx4yO4ePWrqFd3bsS2pePf.jpg?versionId=_biM486XW_7f6m2KWyi8VpfjrERh9EMG", "authoredPostCount": 86, "location": "Email thebeatboxerguy@gmail.com", "blocked": 0, "username": "Marcus Perez", "description": "The beatboxer guy                                                    IG/Twitter/Snapchat \ud83d\udc49 Markaaaay                       LA county", "following": 0, "blocking": 0, "shareUrl": "https://vine.co/MarcusPerez", "profileBackground": "0x333333", "notifyPosts": 0, "followingCount": 86, "repostsEnabled": 0}
location:Email thebeatboxerguy@gmail.com
blocked:0
_attrs:{u'username': u'Marcus Perez', u'followerCount': 1563586, u'vanityUrls': [u'MarcusPerez'], u'userId': 953407442820173824L, u'private': 0, u'likeCount': 1338, u'followingCount': 86, u'postCount': 96, u'explicitContent': 0, u'blocking': 0, u'blocked': 0, u'verified': 0, u'loopCount': 407046259, u'avatarUrl': u'http://v.cdn.vine.co/r/avatars/32750AA2911020482848441946112_pic-r-1386276280332a6588a537d.jpg_jV4ONJF4pmLaHJlxy6Cy5dRT60HXPnnIm718WpzgrSBx4yO4ePWrqFd3bsS2pePf.jpg?versionId=_biM486XW_7f6m2KWyi8VpfjrERh9EMG', u'authoredPostCount': 86, u'description': u'The beatboxer guy                                                    IG/Twitter/Snapchat \U0001f449 Markaaaay                       LA county', u'shareUrl': u'https://vine.co/MarcusPerez', u'profileBackground': u'0x333333', u'notifyPosts': 0, u'location': u'Email thebeatboxerguy@gmail.com', u'following': 0, u'repostsEnabled': 0}
username:Marcus Perez
vanityUrls:[u'MarcusPerez']
blocking:0
name:Marcus Perez
shareUrl:https://vine.co/MarcusPerez
profileBackground:0x333333
notifyPosts:0
followingCount:86
repostsEnabled:0
"""


class UserBio:
    def __init__(self,followerCount,userId,private,likeCount,postCount,explicitContent,description,verified,
                 loopCount,authoredPostCount,location,name,username,shareUrl,followingCount,repostsEnabled):
        self.followerCount = followerCount
        self.userId = userId
        self.private = private
        self.likeCount = likeCount
        self.postCount = postCount
        self.explicitContent = explicitContent
        self.description = description
        self.verified = verified
        self.loopCount = loopCount
        self.authoredPostCount = authoredPostCount
        self.location = location
        self.name = name
        self.username = username
        self.shareUrl = shareUrl
        self.followingCount = followingCount
        self.repostsEnabled = repostsEnabled
        self.usernumber = 0

def getBioOfUser(vineList,userid,apiCount,limit):
    vine = vineList[apiCount]
    bio = None
    try:
        user = vine.get_user(user_id=userid,size='1')
        followerCount= str(user['followerCount'])
        userId= str(user['userId'])
        private= str(user['private'])
        likeCount= str(user['likeCount'])
        postCount= str(user['postCount'])
        explicitContent= str(user['explicitContent'])
        try:
            description= str(user['description'].encode('ascii', 'ignore').decode('ascii'))
        except Exception as e:
            try:
                description= str(user['description'])
            except Exception:
                description = ""
        verified= str(user['verified'])
        loopCount= str(user['loopCount'])
        authoredPostCount= str(user['authoredPostCount'])
        try:
            location= str(user['location'].encode('ascii', 'ignore').decode('ascii'))
        except Exception as e:
            try:
                location= str(user['location'])
            except Exception:
                location = ""
        try:
            name= str(user['name'].encode('ascii', 'ignore').decode('ascii'))
        except Exception as e:
            try:
                name= str(user['name'])
            except Exception:
                name = ""
        try:
            username= str(user['username'].encode('ascii', 'ignore').decode('ascii'))
        except Exception as e:
            try:
                username= str(user['username'])
            except Exception:
                username = ""
        shareUrl= str(user['shareUrl'])
        followingCount= str(user['followingCount'])
        repostsEnabled= str(user['repostsEnabled'])
        bio = UserBio(followerCount,userId,private,likeCount,postCount,explicitContent,description,verified,
                     loopCount,authoredPostCount,location,name,username,shareUrl,followingCount,repostsEnabled)
        return (bio,apiCount)
    except Exception as e:
        if "permission" in str(e):
            print "private user."
            return bio
        if "try again later" in str(e): 
            apiCount = apiCount + 1
            apiCount = apiCount % limit
            vine = vineList[apiCount]
            print "swicthing api. now api is :"+str(apiCount)+": exception"+str(e)
            try:
                user = vine.get_user(user_id=userid,size='1')
                followerCount= str(user['followerCount'])
                userId= str(user['userId'])
                private= str(user['private'])
                likeCount= str(user['likeCount'])
                postCount= str(user['postCount'])
                explicitContent= str(user['explicitContent'])
                try:
                    description= str(user['description'].encode('ascii', 'ignore').decode('ascii'))
                except Exception as e:
                    try:
                        description= str(user['description'])
                    except Exception:
                        description = ""
                verified= str(user['verified'])
                loopCount= str(user['loopCount'])
                authoredPostCount= str(user['authoredPostCount'])
                try:
                    location= str(user['location'].encode('ascii', 'ignore').decode('ascii'))
                except Exception as e:
                    try:
                        location= str(user['location'])
                    except Exception:
                        location = ""
                try:
                    name= str(user['name'].encode('ascii', 'ignore').decode('ascii'))
                except Exception as e:
                    try:
                        name= str(user['name'])
                    except Exception:
                        name = ""
                try:
                    username= str(user['username'].encode('ascii', 'ignore').decode('ascii'))
                except Exception as e:
                    try:
                        username= str(user['username'])
                    except Exception:
                        username = ""
                shareUrl= str(user['shareUrl'])
                followingCount= str(user['followingCount'])
                repostsEnabled= str(user['repostsEnabled'])
                bio = UserBio(followerCount,userId,private,likeCount,postCount,explicitContent,description,verified,
                             loopCount,authoredPostCount,location,name,username,shareUrl,followingCount,repostsEnabled)
                return (bio,apiCount)
            except Exception as ex2:
                print str(ex2)
                return (None,apiCount)
        else:
            return None