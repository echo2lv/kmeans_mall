"""
Kmeans Python learning file
Kaggle reference: https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis

changelog:
    LV Added additional column 'Credit Score' using XLS random number generator between 600 - 850
"""

#libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import seaborn as sns
from sklearn.cluster import KMeans

#create dataframe
df = pd.read_csv(r'C:/Users/Jari/Desktop/dev/kmeans_mall/Mall_Customers_mvar.csv')

"""
#multi variable kmeans; id number of clusters... look for elbow

inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='full') )
    algorithm.fit(X3)
    inertia.append(algorithm.inertia_)

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()
"""

#run kmeans using elbow = 3
X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)', 'Credit Score']].iloc[: , :].values
algorithm = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

#push label to original dataframe
df['cluster_grp'] = pd.Series(algorithm.labels_, index=df.index)
df.to_csv(r'C:/Users/Jari/Desktop/dev/kmeans_mall/Mall_Customers_mvar_cluster.csv')