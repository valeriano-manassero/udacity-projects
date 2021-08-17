from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy2


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_folder_path = config["output_folder_path"]

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for f in [
        "ingestedfiles.txt", "trainedmodel.pkl", "encoder.pkl", "latestscore.txt", ]:
        if f in ["ingestedfiles.txt"]:
            source_filepath = os.path.join(output_folder_path, f)
        else:
            source_filepath = os.path.join(model_path, f)
        new_filepath = os.path.join(prod_deployment_path, f)
        print(f'Copying {source_filepath} to {new_filepath}')
        copy2(source_filepath, new_filepath)


if __name__ == '__main__':
    store_model_into_pickle()
