import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt

K = 30

print('Reading Brightkite_totalCheckins.txt')

#filepath = 'example.txt'
filepath = 'Brightkite_totalCheckins.txt'
checkins = []  
with open(filepath) as fp:  
	line = fp.readline()
	while line:
		line = fp.readline()
		checkins.append(line.strip().split('\t'))
		#print('line : ',line)

length = len(checkins)
del checkins[length-1]
length = len(checkins)

#print('checkins : ')
#print(checkins)

print('Finished Reading Brightkite_totalCheckins.txt')

print('length : ', length)

#Creating data points
DataPoints = []
 

#List of all unique userIds 
userIds = []

for checkin in checkins :
	#print('checkin : ', checkin)
	if(len(checkin)==5) :
		temp = []
		temp.append(float(checkin[2]))
		temp.append(float(checkin[3]))
		temp.append(int(checkin[0]))
		temp.append(checkin[1])
		temp.append(checkin[4])
		DataPoints.append(temp)
		userIds.append(int(checkin[0]))

print('Number of user Ids : ')
print(len(userIds))

userIds = list(dict.fromkeys(userIds))
		
print('Number of unique user Ids : ')
print(len(userIds))
 
length = len(DataPoints)

print('DataPoints : ')
print(length)




def compute_euclidean_distance(point, centroid) :
	dist = abs(point[0] - centroid[0]) + abs(point[1] - centroid[1])
	return dist

def assign_label_cluster(distance, data_point, centroids):
	index_of_minimum = min(distance, key=distance.get)
	#print('index : ', index_of_minimum)
	return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(labels, centroids):

	#print('labels : ')
	#print(labels[0])
	k = K
	arr = [[] for i in range(k)]
	for i in range(k) :
		l = [0, 0]
		arr[i].extend(l)
	#print('arr : ', arr)
	#Iterating for each centroid label

	for i in range(len(labels)) :
		#print(' i : ', i)
		Datapts = []
		for dp in labels[i] :
			Datapts.append(dp[1])
		Datapts.append(centroids[i])
		#print('Datapts : ', Datapts)
		N = float(len(Datapts))
		
		for j in range(len(Datapts)) :
			#print(' j : ', j)
			#print('arr[i] : ', arr[i])
			#print('len(arr[i]) : ', len(arr[i]))
			#print('Datapts[j] : ', Datapts[j])
			for x in range(len(arr[i])) :
				arr[i][x] = arr[i][x] + Datapts[j][x]
		for x in range(len(arr[i])) :
			arr[i][x] = arr[i][x] / N

		#print('arr : ', arr)
	newCentroids = arr
	return newCentroids

def k_means(data_points, centroids, iterations):
	label = []
	cluster_label = []
	total_points = len(data_points)
	k = len(centroids)
    
	for iteration in range(0, iterations):
	    
		print('iteration : ', iteration)

		#Stores the labels and the data point of each data point 
		labels = [[] for i in range(k)]
		for index_point in range(0, total_points):
#			print('Data point : ', index_point)
		
			distance = {}
			for index_centroid in range(0, k):
				#print("data_points[index_point] : ", data_points[index_point])
				#print("centroids[index_centroid] : ", centroids[index_centroid])
				distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
			label = assign_label_cluster(distance, data_points[index_point], centroids)
			labels[label[0]].append(label)
			#print('label : ')
			if iteration == (iterations - 1):
				cluster_label.append(label)
		#print('len : ', len(labels))

		centroids = compute_new_centroids(labels, centroids)

			

	return [cluster_label, centroids]

def print_label_data(result):
	print("Result of k-Means Clustering: \n")
	
	count = []
	for i in range(K) :
		count.append(0)

	for data in result[0]:
		count[data[0]] = count[data[0]] + 1
		print("data point: {}".format(data[1]))
		print("cluster centroid: {} \n".format(data[0]))
	
	print('count : ',count)
	print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
	centroids = []
	indexes = []
	"""
	seen = []
	UniqueDataPoints  = []
	for item in DataPoints:
		if item not in seen:
			seen.append(item)
			UniqueDataPoints.append(item)
	"""
    
	indexes = random.sample(range(0,len(DataPoints)), K)
	
	for index in indexes :
		centroids.append(DataPoints[index])
	print('number of centroids : ', len(centroids))
	#print('centroids : ', centroids)
	return centroids



centroids = create_centroids()
iterations = 5
    
[cluster_label, new_centroids] = k_means(DataPoints, centroids, iterations)
#print_label_data([cluster_label)

x = []
y = []
Clusters = []

for i in range(len(cluster_label)) :
	Cl = cluster_label[i]
	Dp = Cl[1]
	x.append(Dp[0])
	y.append(Dp[1])
	Clusters.append(Cl[0])
centers = new_centroids


def plot_graph(title, x, y, Clusters):
	plt.title(title)
	plt.scatter(x, y, c=Clusters)
	plt.xlabel("Latitudes")
	plt.ylabel("Longitudes")
	plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.8))
	plt.savefig('KMeans.png')

#plot_graph('Kmeans', x, y, Clusters)

clusterIndexes = [[] for i in range(K)]
userIdsClusters = [[] for i in range(K)]


print('Storing userIds present in each cluster')
for data in cluster_label:
	clusterIndexes[data[0]].append(data[1])
	Dp =  data[1]
	userIdsClusters[data[0]].append(Dp[2])

for i in range(K) :
	userIdsClusters[i] = list(dict.fromkeys(userIdsClusters[i]))

def aprioriGen(L1, k) :
	C = []
	
	#Lk-1
	L = L1[k]

	if(k==0) :
		for i in range(len(L)) : 
			for j in range(i+1,len(L)) :
				common = []
				if(L[i]!=L[j]) :
					common.append(L[i])
					common.append(L[j])
					common = sorted(common)
					C.append(common)
		return C	

	for i in range(len(L)) : 
		for j in range(i+1,len(L)) :

			common = []
			diff = []
			
			set1 = set(L[i])
			set2 = set(L[j])
			set3 = set1.union(set2)
			
			length = len(set3)
	
			if(length==(k+1)) : 	 
				common = list(set3)
				common = sorted(common)
				C.append(common)
			
	return C

#Minsupport
minsup = 10

i = 0

#For storing the support count of each author 
countMap = dict()

#Apriori algorithm 

#Frequent itemset list
L = []

#Candidate itemset list
C = []


#Frequent itemset list of length 1
li = []

#Starting the timer for the running of Apriori algorithm
start = time. time()

print('Finding frequent 1-itemsets')
#Finding frequent 1-itemsets

for userId in userIds :

	countMap[userId] = 0

	for index in range(len(userIdsClusters)) : 

		if (userId in userIdsClusters[index]) :

			#To avoid repeated addition of the same element
			if(countMap[userId] != -1) :
				countMap[userId] = countMap[userId] + 1
				if(countMap[userId]>=minsup) : 
					li.append(userId)
					countMap[userId] = -1

#Appending frequent 1-itemsets to frequent itemset list
L.append(li)

f= open("FrequentItemsets.txt","w+")

k = 0

for ele2 in li :
        f.write("%s\n" % ele2)

f.write("\n")

f.write("k = %d \n" % k)
length1 = len(li)
print("Length of set : \n", length1)
f.write("Length of set : %d \n" %(length1))
f.write("\n")


#Apriori starts .. 

C = []


index = 3
print('Limit of k : ', index)

for k in range(1, index) :

	print('Apriori running for k = ', k)
	
	if(len(L)!=k) : 
		break
	
	print('Generating candidate sets :') 
	C = aprioriGen(L, k-1)
	print('Candidate sets generated:') 

	#For creating the hash map of candidate itemsets to support count
	mapping = dict()
	
	for c in C :
		c = frozenset(c)
		mapping[c] = 0

	for ind in range(len(userIdsClusters)) :
		clusterSet = set(userIdsClusters[ind])
		for c in C :
			fc = frozenset(c)
			setC = set(c)
			
			#print("Candidate set : \n")
			#print(c)
	
			#print("Author set : \n")
			#print(authors)
		
			if(setC.issubset(clusterSet)) : 
				mapping[fc] = mapping[fc]+1;	
			
	List = []

	for c in C :
		fc = frozenset(c)
		if(mapping[fc]>=minsup) :	
			List.append(c)

	if List :
		L.append(List)

#Stopping the timer for the running of Apriori algorithm
end = time. time()

print("Sets  of  2  or  more  authors  who  have  frequently  co-authored  articles using Apriori Algorithm:\n") 

count = 0

k = 0

for ele in L : 

	if (k>=1) : 

		for ele2 in ele :
			
			f.write("%s\n" % ele2)

		f.write("\n")

	f.write("k = %d \n" % k)
	length1 = len(ele)
	f.write("Length of set : %d \n" %(length1))
	f.write("\n")

	count = count + len(ele)
	
	k = k+1


print("Length of 2 or more length frequent itemset : ")
print(count)
print("\n")
	
	

print("Time taken for running of Apriori algorithm")
print(end - start)





