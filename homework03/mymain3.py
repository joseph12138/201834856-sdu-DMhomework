import numpy as np
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.cluster import normalized_mutual_info_score
import json


text = []
tags = []

def openfile():
	##BEGIN
	##打开Tweets.txt文件
	with open('Tweets.txt') as file:
		lines = file.read().split('\n')[:-1]
		for line in lines:
			line = json.loads(str(line))
			text.append(line['text'].split())
			tags.append(line['cluster'])
	##END

def kmeans(data, tags):
	##BEGIN
	##使用K-Means算法
	prediction = KMeans(n_clusters = 2472).fit_predict(data)  
	rel = metrics.normalized_mutual_info_score(tags,prediction)
	print("1.K-Means算法的准确率为: " +rel)
	##END

def affinity_propagation(data, tags):
	##BEGIN
	##使用Affinity propagation算法
	prediction = AffinityPropagation(damping=0.75, max_iter=468, convergence_iter=686, copy=True, preference=None, affinity='euclidean', verbose=False).fit_predict(data)
	rel = metrics.normalized_mutual_info_score(tags, prediction)
	print("2.Affinity propagation算法的准确率为:" + rel)
	##END

def mean_shift(data, tags):
	##BEGIN
	##使用Mean-shift算法
	prediction = MeanShift(bandwidth=0.75, bin_seeding=True).fit_predict(data)
	rel = metrics.normalized_mutual_info_score(tags, prediction)
	print("3.Mean-shift算法的准确率为:" + rel)
	##END

def spectral_clustering(data, tags):
	##BEGIN
	##使用Spectral clustering算法
	prediction = SpectralClustering(n_clusters=2472, affinity='nearest_neighbors', n_neighbors=4, eigen_solver='arpack', n_jobs=1).fit_predict(data)	
	rel = metrics.normalized_mutual_info_score(tags, prediction)
	print("4.Spectral clustering算法的准确率为:" + rel)
	##END

def agglomerative_clustering(data, tags):
	##BEGIN
	##使用Agglomerative clustering算法
	prediction = AgglomerativeClustering(n_clusters=2472, affinity='euclidean', linkage='ward').fit_predict(data)
	rel = metrics.normalized_mutual_info_score(tags, prediction)
	print("5.Agglomerative clustering算法的准确率为:" + rel)
	##END

def dbscan(data, tags):
	##BEGIN
	##使用DBSCAN算法
	prediction = DBSCAN(eps=0.75, min_samples=1).fit_predict(data)
	rel = metrics.normalized_mutual_info_score(tags, prediction)
	print("6.DBSCAN算法的准确率为:" + rel)
	##END

def gaussian_mixtures(data, tags):
	##BEGIN
	##使用Gaussian mixtures算法
	prediction = GaussianMixture(n_components=2472, covariance_type='diag', max_iter=20, random_state=0).fit_predict(data)
	el = metrics.normalized_mutual_info_score(tags, prediction)
	print("7.Gaussian mixtures算法的准确率为:" + rel)
	##END

if __name__ == '__main__':
	openfile()                             ##打开词典文件，将文本保存为text，将类别保存到tags
	kmeans(text,tags)                      
	affinity_propagation(text,tags)
	mean_shift(text,tags)
	spectral_clustering(text,tags)
	agglomerative_clustering(text,tags)
	dbscan(text,tags)
	gaussian_mixtures(text,tags)
	
	