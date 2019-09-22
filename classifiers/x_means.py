from pyclustering.cluster.xmeans import xmeans 
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.base import BaseEstimator

class X_means(BaseEstimator):
	"""docstring for X_means"""
	def __init__(self,min_clusters, max_clusters):
		
		self.min_clusters = min_clusters
		self.max_clusters = max_clusters

		
		

	def fit(self,X,y=None):
		
		initial_number_of_cluster_centers = self.min_clusters

		initial_centers = kmeans_plusplus_initializer(X, initial_number_of_cluster_centers).initialize()

		self.xmeans_instance = xmeans(X,initial_centers,self.max_clusters)

		self.xmeans_instance.process()


		return self
	

	def predict(self, x, y=None):
		predicted_cluster = self.xmeans_instance.predict(x)

		return predicted_cluster

		