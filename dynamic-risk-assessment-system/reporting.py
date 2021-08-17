import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    y_pred, df_y = model_predictions(None)
    df_cm = metrics.confusion_matrix(df_y, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(df_cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            ax.text(x=j, y=i,s=df_cm[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(model_path, "confusionmatrix.png"))


if __name__ == "__main__":
    score_model()
