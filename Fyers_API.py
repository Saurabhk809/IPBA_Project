App_ID="JBCBGH3ANZ-100"
Secret_ID="P8D5F8OR2U"

redirect_url="https://www.google.com/"

# Authorisation
from fyers_api import fyersModel
from fyers_api import accessToken
from fyers_api import accessToken
import os

# generate a session
def login ():
    if not os.path.exists("acess_token.txt"):
        session=accessToken.SessionModel(client_id=App_ID,secret_key=Secret_ID,redirect_uri=redirect_url,response_type="code")
        response = session.generate_authcode()
        print("Login URL:",response)
        auth_code=input("Enter Auth Code : ")
        session.set_token(auth_code)
        access_token = session.generate_token()
        access_token=session.generate_token()['access_token']
        with open ("access_token.txt","w") as f:
            f.write(accessToken)
    else:
            with open("access_token.txt","r") as f:
                access_token=f.read()
    return access_token
    #print (response)

print(login())
