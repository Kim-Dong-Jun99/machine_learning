# importing dataset and cluster algorithm to use
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# loading iris dataset into iris
iris = datasets.load_iris()

# we have to modify the dataset by removing the id, and species attribute
# to check about our dataset, lets check for description
print(iris.DESCR)

print(iris.data)

print(iris.target)

print(iris.feature_names)

print(iris.target_names)

# separating data into data and target, denoting X as data, Y as target
X = iris.data
y = iris.target

# since we have to use 4 clusters, denoted n_clusters=4, and denoted random_state=42 to have same result
km = KMeans(n_clusters=4, random_state=42)

# to get silhouette, we must fit and predict modified dataset with k-means model
km.fit_predict(X)

# calculate silhouette_score
score = silhouette_score(X, km.labels_, metric='euclidean')
print(score)

# specified range of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
# list for saving silhouette scores
silhouette_avg = []

for num in range_n_clusters:
    kmeans = KMeans(n_clusters=num)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    # calculate silhouette score for each cluster number and save it in the list
    silhouette_avg.append(silhouette_score(X, cluster_labels))

# plot silhouette scores for each cluster numbers
plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel("Values of K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette analysis For optimal K")
plt.show()

