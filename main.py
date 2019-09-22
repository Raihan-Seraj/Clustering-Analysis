import numpy as np
import json 
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests
import preprocessing 
import classification

with open('configurations/configs_Cleaveland.json') as config_file:
    configs = json.load(config_file)


#data_name= 'Wisconsin_Diagnostic_Breast_Cancer'
data_name = 'Cleaveland_Heart_Disease'
#data_name = 'Credit_Approval'
#data_name = 'Epileptic_Seizure'
#data_name = 'KDD_10_percent'
data_url = configs['datasets'][data_name]


features, labels = preprocessing.preprocess(data_url,data_name,configs)

scores = classification.classify(features,labels,configs)

print("The accuracy scores for 5 folds are, ",scores[0])
print("The adjusted rand score for 5 folds are, ",scores[1])
print("The homogeneity scores for 5 folds are, ",scores[2])
print("The mutual information scores for 5 folds are, ",scores[3])
print("The completeness scores for 5 folds are, ",scores[4])


if configs['classification type']=="Binary":
	print("The f1 scores for 5 folds are, ",scores[5])





