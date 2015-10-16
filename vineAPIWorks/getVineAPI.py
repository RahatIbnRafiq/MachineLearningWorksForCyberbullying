'''
Created on Jan 20, 2015

@author: RahatIbnRafiq
'''

import vinepy
import getAPIKeys

def getVineAPI():
    apis = getAPIKeys.getApiList()
    vineList = []
    for api in apis:
        vine = vinepy.API(username=api.username, password=api.password,session_id=api.sessionid)
        vineList.append(vine)
    print "API Collection is Done"
    return vineList
    
    """password = ""
    f = open("password.txt","r")
    for line in f:
        password = line.strip()
        break
    f.close()
        
    
    vine = vinepy.API(username='rahatibnrafiq@gmail.com', password=password)
    return vine"""
