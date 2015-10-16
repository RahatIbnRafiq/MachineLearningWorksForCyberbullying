'''
Created on Jan 20, 2015

@author: RahatIbnRafiq
'''





def getBioOfUser(vineList,userid,apiCount):
    vine = vineList[apiCount]
    try:
        user = vine.get_user(user_id=userid,size='1')
        followerCount = str(user["followerCount"])
        followingCount = str(user["followingCount"])
        return (followerCount,followingCount,apiCount)
    except Exception as e:
        print str(e)
        if "permission" in str(e):
            print "private user"
            return None
        if "followerCount" in str(e):
            print str(e)
            return None
        else:
            print str(e)
            print "try again later exception"
            apiCount = apiCount + 1
            apiCount = apiCount % 4
            vine = vineList[apiCount]
            print "Switching API. now API count is "+str(apiCount)
            try:
                user = vine.get_user(user_id=userid,size='1')
                followerCount = str(user["followerCount"])
                followingCount = str(user["followingCount"])
                return (followerCount,followingCount,apiCount)
            except Exception as e:
                print str(e)
                print "this is the final exception in getBioOfUser function."
                return None
        