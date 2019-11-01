
'''
Created by Raihan
'''
import requests
import io
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import category_encoders as ce
import glob
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def preprocess(data_url, dataname,configs): 
	s=requests.get(data_url).content
	
	if dataname=="KDD_10_percent":
		
		data = pd.read_csv('./Kdd_Data/kddcup.data_10_percent')
	elif dataname=="UNSW":
		file_location=glob.glob("./UNSW_data/unsw_data.csv")
		li=[]
		for filenames in file_location:
			data = pd.read_csv(filenames, low_memory=False) 

			# dtypes={"srcip": object, "sport": int, "dstip": object, "dsport": int, "proto": object, "state": object, 
			# "dur": float, "sbytes": int, "dbytes": int, "sttl": int, "dttl": int, "sloss": int, "dloss": int, "service": object, 
			# "Sload": float, "Dload": float, "Spkts": int, "Dpkts": int, "swin": int, "dwin": int, "stcpb": int, 
			# "dtcpb": int, "smeansz": int, "dmeansz": int, "trans_depth": int, "res_bdy_len": int, "Sjit": float,
			# "Djit": float, "Sintpkt": float, "Dintpkt": float, "tcprtt": float, "synack": float, "ackdat": float,
			# "is_sm_ips_ports": int, "ct_state_ttl": int, "ct_flw_http_mthd": int, "is_ftp_login": int, "ct_ftp_cmd": int, 
			# "ct_srv_src": int, "ct_srv_dst": int, "ct_dst_ltm": int, "ct_src_ltm": int, "ct_src_dport_ltm": int, 
			# "ct_dst_sport_ltm": int, "ct_dst_src_ltm": int, "attack_cat": object, "Label": int, "Stime": int, "Ltime": int }

			# df.columns=['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
			# 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
			# 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
			# 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
			# 'ct_dst_src_ltm', 'attack_cat', 'Label']

			# df.astype(dtypes)

		
		#import ipdb;ipdb.set_trace()
	else:

		data = pd.read_csv(io.StringIO(s.decode('utf-8')))

	
	if dataname=="Wisconsin_Diagnostic_Breast_Cancer":

		y_train = np.array(pd.factorize(data.iloc[:,1])[0]).reshape(-1,1)
		
		features=np.array(data.iloc[:,2:-1])
		
		#import ipdb;ipdb.set_trace()

		if configs['preprocessing']['Missing_Instance']:
			print("Missing Instance replaced with median")

			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			
			features_new = imp.fit_transform(features)
			
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)
		else:
			features = features

		
		if configs['preprocessing']['StandardScaler']:
			print("Using Standard Scaler")
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
			print("Missing Instance replaced with median")
			#using mean-strategies to replace the missing values
			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			features_new = imp.fit_transform(features)
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)
		else:
			features = feautures

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
			print("Using Standard Scaler")
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features

		elif configs['preprocessing']['MinMaxScaler']:
			from sklearn.preprocessing import MinMaxScaler
			print("Using MinMax Scaler")
			
			scaler=MinMaxScaler(feature_range=(0,1))
			
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
			print("Missing instance replaced with median")
			imp = SimpleImputer(missing_values='?',fill_value=np.nan, strategy='constant')
			features_new = imp.fit_transform(features)
			imp_rem = SimpleImputer(missing_values=np.nan, strategy='median')

			features = imp_rem.fit_transform(features_new)


		else:
			features = features

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
			print("Using Standard Scaler")
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features

		elif configs['preprocessing']['MinMaxScaler']:
			from sklearn.preprocessing import MinMaxScaler
			print("Using MinMax Scaler")
			
			scaler=MinMaxScaler(feature_range=(0,1))
			
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else: 
			X_train = features
		
	if dataname=="Epileptic_Seizure":
		y_train = np.array(data.iloc[:,-1])
		features = np.array(data.iloc[:,1:-1])

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
			print("Using StandardScaler")
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		elif configs['preprocessing']['MinMaxScaler']:
			from sklearn.preprocessing import MinMaxScaler
			print("Using MinMax Scaler")
			
			scaler=MinMaxScaler(feature_range=(0,1))
			
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else:
			X_train = features

	if dataname=="KDD_10_percent":
		y_train = pd.factorize(data.iloc[:,-1])[0]
		
		features = data.iloc[:,0:-1]

		features.iloc[:,1] = pd.factorize(data.iloc[:,1])[0]

		features.iloc[:,2] = pd.factorize(data.iloc[:,2])[0]

		features.iloc[:,3] = pd.factorize(data.iloc[:,3])[0]

		features = np.array(features)

	

		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
			print("Using StandardScaler")
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		elif configs['preprocessing']['MinMaxScaler']:
			from sklearn.preprocessing import MinMaxScaler
			print("Using MinMax Scaler")
			
			scaler=MinMaxScaler(feature_range=(0,1))
			
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else:
			X_train = features

	if dataname=="UNSW":
		
		dataset=clean_dataset(data)
		#import ipdb;ipdb.set_trace()
		y_train=np.array(dataset.iloc[:,-1])
		
		# import ipdb;ipdb.set_trace()
		# for i in range(features.shape[1]):
		# 	features.iloc[:,i]=pd.to_numeric(features.iloc[:,i], errors='coerce')
		# import ipdb;ipdb.set_trace()
		features= np.array(dataset.iloc[:,0:-1])
		
		if configs['preprocessing']['StandardScaler']:
			from sklearn.preprocessing import StandardScaler
			print("Using StandardScaler")
		
			scaler = StandardScaler()
		
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		elif configs['preprocessing']['MinMaxScaler']:
			from sklearn.preprocessing import MinMaxScaler
			print("Using MinMax Scaler")
			
			scaler=MinMaxScaler(feature_range=(0,1))
			
			scaled_features = scaler.fit_transform(features)

			X_train = scaled_features
		else:
			X_train = features


	return X_train, y_train.reshape(-1,) 






