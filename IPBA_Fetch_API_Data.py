import requests
import pandas as pd
import time

#keys
#API_Key
#ID


# Make API Call
#response=requests.get('location # URL')

# Create an API request
url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'
response = requests.get(url)
print("Status code: ", response.status_code)
# In a variable, save the API response.
response_dict = response.json()
# Evaluate the results.
print(response_dict.keys())