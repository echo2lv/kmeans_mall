"""
Kmeans Python learning file
Kaggle reference: https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis

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
df = pd.read_csv(r'C:/Users/Jari/Desktop/dev/kmeans_mall/Mall_Customers.csv')

#explore data
df.head()
df.shape
df.describe()
df.dtypes
#search for nulls
df.isnull().sum()

#visualize the data 
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.histplot(
        df[x],
        #kind='ecdf'
        kde=True,
        bins = 20
        )
    plt.title('Distplot of {}'.format(x))
plt.show()

#k-means; select clusters based on visible elbow
x1 = df[['Age', 'Spending Score (1-100)']].iloc[:,:].values
inertia = []
for n in range (1,11):
    alg = (KMeans(n_clusters= n, init='k-means++', n_init= 10, max_iter= 300,
            tol=0.0001, random_state= 111, algorithm='full'))
    alg.fit(x1)
    inertia.append(alg.inertia_)
    
#identify # of clusters needed
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

#kmeans using N value
alg = (KMeans(n_clusters= 4, init='k-means++', n_init= 10, max_iter= 300,
            tol=0.0001, random_state= 111, algorithm='full'))
alg.fit(x1)
labels1 = alg.labels_
centroids1 = alg.cluster_centers_

#viz results
h = 0.02
x_min, x_max = x1[:, 0].min() - 1, x1[:, 0].max() + 1
y_min, y_max = x1[:, 1].min() - 1, x1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = alg.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()

#push label to original dataframe
df['age_spend_cluster'] = pd.Series(alg.labels_, index=df.index)
#df.to_csv(r'C:/Users/Jari/Desktop/dev/kmeans_mall/Mall_Customers_cluster.csv')

#multi variable kmeans; id number of clusters... look for elbow
X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
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

#run kmeans using elbow = 6
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

df['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['label3'], 
        size= 20,
        line=dict(
            color= df['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)

#push label to original dataframe
df['age_spend_income_cluster'] = pd.Series(algorithm.labels_, index=df.index)
df.to_csv(r'C:/Users/Jari/Desktop/dev/kmeans_mall/Mall_Customers_cluster.csv')