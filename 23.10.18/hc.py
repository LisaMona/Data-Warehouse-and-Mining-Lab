import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('iris.csv')

#X = dataset.iloc[:, :-1]
X = dataset.iloc[:, [0,1, 2, 3]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'single'))
#d = sch.dendrogram(sch.linkage())
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
#s = AgglomerativeClustering()
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'orange', label = 'Iris-setosa')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Iris-virginica')
plt.legend()
