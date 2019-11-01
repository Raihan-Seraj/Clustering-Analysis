from sklearn.cluster import *
# from pyclustering.cluster.xmeans import xmeans 
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from classifiers import x_means
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
from kmodes.kprototypes import KPrototypes
from k_means_constrained import KMeansConstrained
from sklearn.pipeline import Pipeline

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

		min_num_constraints = configs['algorithm']['minimum number of constraints']

		max_num_constraints = configs['algorithm']['maximum number of constraints']

		num_clusters = configs['algorithm']['number of clusters']



		clf = KMeansConstrained(n_clusters=num_clusters, size_min=min_num_constraints, size_max=max_num_constraints)

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

		num_clusters=configs['algorithm']['number of clusters']

		clf=KPrototypes(n_clusters=num_clusters,init='Huang')
		fit_params={'categorical': [1,2]}

		accuracy_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='accuracy',fit_params=fit_params)

		rand_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='adjusted_rand_score',fit_params=fit_params)

		homogeneity_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='homogeneity_score',fit_params=fit_params)

		mutual_info_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='mutual_info_score',fit_params=fit_params)

		completeness_scores = cross_val_score(clf, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='completeness_score',fit_params=fit_params)

		if configs['classification type']=="Binary":

			f1_scores = cross_val_score(pipeline, X_train, y_train, cv=configs['preprocessing']['Cross_Validation_Fold'],scoring='f1_macro', fit_params=fit_params)

			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores,f1_scores
		else:
			return accuracy_scores, rand_scores, homogeneity_scores, mutual_info_scores, completeness_scores

		
	
	if configs['algorithm']['name']=='Kernel_K-means':
		pass

		



