
'''
Created by Raihan
'''
import requests
import io
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
def preprocess(data_url, dataname,configs): 
	s=requests.get(data_url).content
	data = pd.read_csv(io.StringIO(s.decode('utf-8')))

	if dataname=="Wisconsin_Diagnostic_Breast_Cancer":

		y_train = np.array(pd.factorize(data.iloc[:,1])[0]).reshape(-1,1)
		
		features=np.array(data.iloc[:,2:-1])
		
		#import ipdb;ipdb.set_trace()

		if configs['preprocessing']['Missing_Instance']:

			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			
			features_new = imp.fit_transform(features)
			
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)
		else:
			features = features

		
		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else:

			X_train = features



	if dataname=="Cleaveland_Heart_Disease":

		y_train = np.array(data.iloc[:,-1]).reshape(-1,1)
		features = np.array(data.iloc[:,0:-1])

		if configs['preprocessing']['Missing_Instance']:
			#using mean-strategies to replace the missing values
			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			features_new = imp.fit_transform(features)
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)
		else:
			features = feautures

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else:
			X_train = features



	if dataname=="Credit_Approval":
		y_train = np.array(pd.factorize(data.iloc[:,-1])[0]).reshape(-1,1)
		features = data.iloc[:,0:-1]
		features.iloc[:,0] = pd.factorize(data.iloc[:,0])[0]
		features.iloc[:,3] = pd.factorize(data.iloc[:,3])[0]
		features.iloc[:,4] = pd.factorize(data.iloc[:,4])[0]
		features.iloc[:,5] = pd.factorize(data.iloc[:,5])[0]
		features.iloc[:,6] = pd.factorize(data.iloc[:,6])[0]
		features.iloc[:,8] = pd.factorize(data.iloc[:,8])[0]
		features.iloc[:,9] = pd.factorize(data.iloc[:,9])[0]
		features.iloc[:,11] = pd.factorize(data.iloc[:,11])[0]
		features.iloc[:,12] = pd.factorize(data.iloc[:,12])[0]

		if configs['preprocessing']['Missing_Instance']:
			#using mean-strategies to replace the missing values
			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			features_new = imp.fit_transform(features)
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)


		else:
			features = features

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else: 
			X_train = features
		
		#import ipdb;ipdb.set_trace()
		
	return X_train, y_train.reshape(-1,) 




