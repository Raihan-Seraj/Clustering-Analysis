import numpy as np
import json 
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests
import preprocessing 
import classification

with open('configs_Cleaveland.json') as config_file:
    configs = json.load(config_file)


#data_name= 'Wisconsin_Diagnostic_Breast_Cancer'
data_name = 'Cleaveland_Heart_Disease'
data_url = configs['datasets'][data_name]


features, labels = preprocessing.preprocess(data_url,data_name,configs)

scores = classification.classify(features,labels,configs)


