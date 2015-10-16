'''
Created on Jan 25, 2015

@author: RahatIbnRafiq
'''





class APIKeys:
    def __init__(self,username,password,sessionid):
        self.username = username
        self.password = password
        self.sessionid = sessionid


def getApiList():
    username =""
    password = ""
    sessionid = ""
    apiList = []
    f = open("keys1.txt","r")
    for line in f:
        line =  line.strip()
        data = line.split(",")
        username = str(data[0])
        password = str(data[1])
        sessionid = str(data[2])
        api = APIKeys(username,password,sessionid)
        apiList.append(api)
    f.close()
    return apiList


