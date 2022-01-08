import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame
file=open("points.txt","r")
line=str(file.read()).split("\n") #tutorialspoint.com/numpy/numpy_append.htm
xpoints=[]
ypoints=[]
x, y = str(line[1]).split("\t")
t=np.array([[x,y]])
for i in range(2,len(line)):
   x,y= str(line[i]).split("\t")
   xpoints.append(float(x)*100)
   ypoints.append(float(y)*100)

   t=np.append(t,[[x,y]],axis = 0)




Data = {'x': xpoints,
        'y': ypoints
       }
df = DataFrame(Data,columns=['x','y'])
print (df)
kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
y_kmc = kmeans.fit_predict(df)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()