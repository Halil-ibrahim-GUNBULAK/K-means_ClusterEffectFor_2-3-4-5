import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
file=open("points.txt","r")
line=str(file.read()).split("\n") #tutorialspoint.com/numpy/numpy_append.htm
xpoints=[]
ypoints=[]
x, y = str(line[1]).split("\t")
t=np.array([[x,y]])
for i in range(2,len(line)):
   x,y= str(line[i]).split("\t")

   t=np.append(t,[[x,y]],axis = 0)




print(t)
#plt.scatter(t[:,0],t[:,1], label='True Position')
#plt.show()

nc=2  #choose one  number what you want try for k-means
kmeans = KMeans(n_clusters=nc)
kmeans.fit(t)
print("For n cluster ={}".format(nc))
print("cluster Center:\n",kmeans.cluster_centers_)
print("kmeans Labels:\n",kmeans.labels_)
plt.scatter(t[:, 0], t[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.title("K-Means = {}".format(nc))
plt.show()

plt.title("K-Means = {} avarage points".format(nc),)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.show()

