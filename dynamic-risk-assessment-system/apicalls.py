import requests
import os
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"



#Call each API endpoint and store the responses
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

response1 = requests.post("%s/prediction" % URL, json={"dataset_path": "testdata.csv"}, headers=headers).text
response2 = requests.get("%s/scoring" % URL, headers=headers).text
response3 = requests.get("%s/summarystats" % URL, headers=headers).text
response4 = requests.get("%s/diagnostics" % URL, headers=headers).text


#combine all API responses
responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 
model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as returns_file:
    returns_file.write(responses)
