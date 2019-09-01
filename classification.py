from sklearn.cluster import *
from pyclustering.cluster.xmeans import xmeans 
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score



def classify(X_train, y_train,configs):

	if configs['algorithm']['name']=="K-means":
		print("Running K-Means Algorithm")

		if configs['classification type']=="Binary":
			
			print("Running Binary Classification")

			num_clusters =2
		else:
			print("Running Multiclass Classification")

			num_clusters = configs['algorithm']['number of clusters']

		from sklearn.cluster import KMeans

		clf = KMeans(n_clusters=num_clusters)

		#import ipdb;ipdb.set_trace()
		
		accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy')

		rand_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='adjusted_rand_score')

		homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score')

		mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score')

		completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='completeness_score')

		if configs['classification type']=="Binary":

			f1_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='f1_macro')

			return accuracy_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, homogeneity_scores, mutual_info_scores, completeness_scores

