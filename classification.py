from sklearn.cluster import *
# from pyclustering.cluster.xmeans import xmeans 
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from classifiers import x_means
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
#from kmodes.kprototypes import KPrototypes
from k_means_constrained import KMeansConstrained
from sklearn.pipeline import Pipeline
from classifiers.kmodes.kmodes import kprototypes 
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
def classify(X_train, y_train,configs):

	if configs['algorithm']['name']=="K-means":
		print("Running K-Means Algorithm")

		if configs['classification type']=="Binary":
			
			print("Running Binary Classification")

			num_clusters =2
			#import ipdb;ipdb.set_trace()
		else:
			print("Running Multiclass Classification")

			num_clusters = configs['algorithm']['number of clusters']

		from sklearn.cluster import KMeans

		clf = KMeans(n_clusters=num_clusters)


		
		accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy')

		rand_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='adjusted_rand_score')

		homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score')

		mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score')

		completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='completeness_score')


		if configs['classification type']=="Binary":

			f1_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='f1_macro')

			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores

	if configs['algorithm']['name']=="X-means":

		print("Running X-Means Algorithm")

		
		min_num_clusters = configs['algorithm']['minimum number of clusters']

		max_num_clusters = configs['algorithm']['maximum number of clusters']

		clf = x_means.X_means(min_num_clusters,max_num_clusters)


		

		accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy')

		rand_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='adjusted_rand_score')

		homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score')

		mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score')

		completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='completeness_score')

		if configs['classification type']=="Binary":

			f1_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='f1_macro')

			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores



	if configs['algorithm']['name']=="Constrained_K-means":

		print("Running Constrained K-Means Algorithm")

		min_num_constraints = int(X_train.shape[0]/3)# configs['algorithm']['minimum number of constraints']

		max_num_constraints = None#configs['algorithm']['maximum number of constraints']

		num_clusters = configs['algorithm']['number of clusters']

		#import ipdb;ipdb.set_trace()



		
		clf = KMeansConstrained(n_clusters=num_clusters, size_min=min_num_constraints, size_max=max_num_constraints)

		#clf=KMeansConstrained(n_clusters=num_clusters)

		accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy')

		rand_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='adjusted_rand_score')

		homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score')

		mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score')

		completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='completeness_score')

		if configs['classification type']=="Binary":

			f1_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='f1_macro')

			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores




	
	if configs['algorithm']['name']=='K-Prototype':

		print("Running K-Prototype Algorithm")

		num_clusters=configs['algorithm']['number of clusters']
		#M=configs['algorithm']['Number of features']
		#MN=configs['algorithm']['Number of numerical dimensions']
		categories_col=configs['algorithm']['categorical']
		
		
		catdict={
		"categorical": categories_col
		}
		#import ipdb;ipdb.set_trace()
		clf=kprototypes.KPrototypes(n_clusters=num_clusters,init='Huang')
		#fit_params={'categorical': categorical}
		#hh=np.asanyarray(X_train[:, [ii for ii in range(X_train.shape[1]) if ii not in categories_col]]).astype(float)
		
		#import ipdb;ipdb.set_trace()
		#accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy',fit_params=catdict)

		# for k in range(100):
		# 	clf.fit_predict(X_train, categorical=categories_col)
		# print(" For loop completed")
		accuracy_scores=[]
		rand_scores= []
		homogeneity_scores=[]
		mutual_info_scores=[]
		completeness_scores=[]
		f1_scores=[]

		for i in range(configs['preprocessing']['Cross_Validation_Fold']):
			data = np.concatenate((X_train,y_train.reshape(-1,1)), axis=1)
			X_train_cv, X_test_cv, y_train_cv, y_test_cv=train_test_split(X_train,y_train,test_size=0.33)

			clf.fit(X_train, categorical=categories_col)
			prediction= clf.predict(X_test_cv,categorical=categories_col)


			accuracy_scores.append(metrics.accuracy_score(y_test_cv, prediction))
			rand_scores.append(metrics.adjusted_rand_score(y_test_cv,prediction))
			homogeneity_scores.append(metrics.homogeneity_score(y_test_cv,prediction))
			mutual_info_scores.append(metrics.mutual_info_score(y_test_cv,prediction))
			completeness_scores.append(metrics.completeness_score(y_test_cv,prediction))
			f1_scores.append(metrics.f1_score(y_test_cv,prediction))
			


		# homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score',fit_params=catdict)

		# mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score',fit_params=catdict)

		# completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing'y]['Cross_Validation_Fold'],scoring='completeness_score',fit_params=catdict)

		if configs['classification type']=="Binary":

			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores
		
	
	if configs['algorithm']['name']=='Kernel_K-means':
		pass

		



