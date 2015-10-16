
import unirest


response = unirest.post("https://community-vineapp.p.mashape.com/users/authenticate",
  headers={
    "X-Mashape-Key": "gxShFmp7TCmshoB0O4PP8ya9QQfcp1pKUeBjsnzWJltjpv0o7B",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json"
  },
  params={
    "password": "123456789",
    "username": "c@rahat.com"
  }
)

print response.body["data"]["key"]





"""
import vinepy

username =""
password = ""
sessionid = ""

class APIKeys:
    def __init__(self,username,password,sessionid):
        self.username = username
        self.password = password
        self.sessionid = sessionid


def getApiList():
    apiList = []
    f = open("keys.txt","r")
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


print "done"

"""
