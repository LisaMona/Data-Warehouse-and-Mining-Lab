# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('car.csv')

#assign name to the columns
dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = X.apply(LabelEncoder().fit_transform)
onehotencoder = OneHotEncoder(categorical_features=[2,3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# creating odd list of K for KNN
myList = list(range(1,50))

# empty list that will hold K values
k_values = []

# perform 10-fold cross validation
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_values.append(scores.mean())
    #print "scores\n", scores
    #print "k-values\n", k_values
    #print k_values
    
# changing to misclassification error
MSE = [1 - x for x in k_values]
#AC = [x for x in k_values ]

# determining best k
optimal_k = myList[MSE.index(min(MSE))]
#optimal_k = myList[AC.index(min(AC))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(myList, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn import tree,metrics
count_misclassified = (y_test != y_pred).sum()
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Misclassified samples: {}'.format(count_misclassified))
print("Confusion Matrix: ",cm)
print('Accuracy: {:.2f}'.format(accuracy))
