# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Pick customer segment quantity (k).
2. Seed cluster centers with random data points.
3. Assign customers to closest centers. 
4. Re-center clusters and repeat until stable.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Adchayakiruthika M S
RegisterNumber: 212223230005  

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data

X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
print(labels)

colors = ['r', 'g', 'b', 'c', 'm']

for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()
```

## Output:
## Head:
![image](https://github.com/Adchayakiruthika18/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147139995/c039ab1b-6e86-4141-a66c-412490680cf5)

## X value:
![image](https://github.com/Adchayakiruthika18/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147139995/4a93e342-8cf5-4d31-aa61-b86955ac88c1)

## Plot:
![image](https://github.com/Adchayakiruthika18/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147139995/5f0f2945-bd7a-454e-98c8-0248a4ee3002)

## Centroid and Label:
![image](https://github.com/Adchayakiruthika18/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147139995/1b2a6658-56ac-40d6-a844-ae494452d676)

## K-means clustering:
![image](https://github.com/Adchayakiruthika18/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147139995/fe149193-9719-4b07-b721-245e47a2edbf)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
