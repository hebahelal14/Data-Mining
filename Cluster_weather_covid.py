import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import plotly.express as px
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.cm as cm
import numpy as np

#Dataset
weather_covid = pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Clustering-new\\Weather+Covid-19\\weather_dataset3.csv')

#select only Gulf countries, China, Candinavian Countries, US, Canada, Brazil, and India
weather_covid = weather_covid[(weather_covid['Country/Region'] == 'United Arab Emirates') | (weather_covid['Country/Region'] == 'Bahrain') | (weather_covid['Country/Region'] == 'Qatar') | (weather_covid['Country/Region'] == 'Oman') | (weather_covid['Country/Region'] == 'Saudi Arabia') | (weather_covid['Country/Region'] == 'Kuwait') | (weather_covid['Country/Region'] == 'Iceland')| (weather_covid['Country/Region'] == 'US')| (weather_covid['Country/Region'] == 'Sweden')|(weather_covid['Country/Region'] == 'Denmark')|(weather_covid['Country/Region'] == 'Norway')|(weather_covid['Country/Region'] == 'Finland')|(weather_covid['Country/Region'] == 'Canada')|(weather_covid['Country/Region'] == 'Brazil')|(weather_covid['Country/Region'] == 'India')]
weather_covid = weather_covid.groupby('Country/Region').mean().reset_index()
#weather_covid.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Clustering-new\\Weather+Covid-19\\weather_final.csv')
#print(weather_covid.head())


cluster = weather_covid.drop(['Id','Id.1','Country/Region','deaths','Lat','Long','min','max','stp','slp','dewp','rh','ah','wdsp','prcp','fog'], axis=1)
#cluster.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Clustering-new\\Weather+Covid-19\\cluster.csv')
#print(cluster.head())

#Make Scaling
cluster = StandardScaler().fit_transform(cluster)
print(cluster)

#Silhoutte Analysis to get the number of clusters for KMeans Clustering Algorithm
range_n_clusters = [2, 3, 4, 5]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(cluster) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    #cluster_labels = clusterer.fit_predict(cluster1)
    cluster_labels = clusterer.fit_predict(cluster)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    #silhouette_avg = silhouette_score(cluster1, cluster_labels)
    silhouette_avg = silhouette_score(cluster, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    #sample_silhouette_values = silhouette_samples(cluster1, cluster_labels)
    sample_silhouette_values = silhouette_samples(cluster, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    #ax2.scatter(cluster1[:, 0], cluster1[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #            c=colors, edgecolor='k')

    ax2.scatter(cluster[:, 0], cluster[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')


    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

#KMeans with n_cluster = 2
kmeans = KMeans(n_clusters=4)
kmeans.fit(cluster)
#kmeans.fit(cluster1)
labels = weather_covid['Country/Region']
clusters = kmeans.predict(cluster)
#clusters = kmeans.predict(cluster1)
print("clusters ",clusters)
print(labels)
# assign the clustering labels
weather_covid['cluster'] = clusters
Kmeans = weather_covid.groupby('cluster').mean().reset_index()
#print(Kmeans)
print(weather_covid[weather_covid['cluster'] == 0]['Country/Region'].values.tolist(), end=" ")
print(weather_covid[weather_covid['cluster'] == 1]['Country/Region'].values.tolist(), end=" ")
print(weather_covid[weather_covid['cluster'] == 2]['Country/Region'].values.tolist(), end=" ")
print(weather_covid[weather_covid['cluster'] == 3]['Country/Region'].values.tolist(), end=" ")
#print(weather_covid[weather_covid['cluster'] == 4]['Country/Region'].values.tolist(), end=" ")
#print(weather_covid[weather_covid['cluster'] == 5]['Country/Region'].values.tolist(), end=" ")

#Evaluation
#ari = metrics.cluster.adjusted_rand_score(labels,clusters)
#print("Adjusted Rand Index ",ari)
#nmi = metrics.cluster.normalized_mutual_info_score(labels,clusters)
#print("Normalized Mutual Information ",nmi)
#m = metrics.cluster.adjusted_mutual_info_score(labels,clusters)
#print("adjusted_mutual_info_score ",m)

cluster_labels = kmeans.fit_predict(cluster)
#cluster_labels = kmeans.fit_predict(cluster1)

#silhouette = sm.silhouette_score(cluster1, cluster_labels)
silhouette = sm.silhouette_score(cluster, cluster_labels)
print("The average silhouette_score is :", silhouette)

#accuracy_score = sm.accuracy_score(cluster, cluster_labels)
#print("The accuracy_score is :", accuracy_score)

#Plotting (Before Clustering)
#plt.scatter(cluster1[:, 0],cluster1[:, 1])
plt.scatter(cluster[:, 0],cluster[:, 1])
plt.xlabel("Confirmed Cases")
plt.ylabel("Mean Temperature of the day")
plt.title("The data before Clustering")
plt.show()

center = kmeans.cluster_centers_
print(center)
print(len(cluster_labels))

plt.scatter(center[0][0],center[0][1],marker = '*',s=200,color='y')
plt.scatter(center[1][0],center[1][1],marker = '*',s=200,color='y')
plt.scatter(center[2][0],center[2][1],marker = '*',s=200,color='y')
plt.scatter(center[3][0],center[3][1],marker = '*',s=200,color='y')
#plt.scatter(center[4][0],center[4][1],marker = '*',s=200,color='y')
#plt.scatter(center[5][0],center[5][1],marker = '*',s=200,color='y')


plt.scatter(cluster[cluster_labels==0,0],cluster[cluster_labels==0,1],s=50,color='r', label = 'Cluster 1')
plt.scatter(cluster[cluster_labels==1,0],cluster[cluster_labels==1,1],s=50,color='g', label = 'Cluster 2')
plt.scatter(cluster[cluster_labels==2,0],cluster[cluster_labels==2,1],s=50,color='b', label = 'Cluster 3')
plt.scatter(cluster[cluster_labels==3,0],cluster[cluster_labels==3,1],s=50,color='c', label = 'Cluster 4')
#plt.scatter(cluster[cluster_labels==4,0],cluster[cluster_labels==4,1],s=50,color='m', label = 'Cluster 5')
#plt.scatter(cluster[cluster_labels==5,0],cluster[cluster_labels==5,1],s=50,color='k', label = 'Cluster 6')

plt.text(cluster[0, 0],cluster[0, 1],labels[0],fontsize=8)
plt.text(cluster[1, 0],cluster[1, 1],labels[1],fontsize=8)
plt.text(cluster[2, 0],cluster[2, 1],labels[2],fontsize=8)
plt.text(cluster[3, 0],cluster[3, 1],labels[3],fontsize=8)
plt.text(cluster[4, 0],cluster[4, 1],labels[4],fontsize=8)
plt.text(cluster[5, 0],cluster[5, 1],labels[5],fontsize=8)
plt.text(cluster[6, 0],cluster[6, 1],labels[6],fontsize=8)
plt.text(cluster[7, 0],cluster[7, 1],labels[7],fontsize=8)
plt.text(cluster[8, 0],cluster[8, 1],labels[8],fontsize=8)
plt.text(cluster[9, 0],cluster[9, 1],labels[9],fontsize=8)
plt.text(cluster[10, 0],cluster[10, 1],labels[10],fontsize=8)
plt.text(cluster[11, 0],cluster[11, 1],labels[11],fontsize=8)
plt.text(cluster[12, 0],cluster[12, 1],labels[12],fontsize=8)
plt.text(cluster[13, 0],cluster[13, 1],labels[13],fontsize=8)
plt.text(cluster[14, 0],cluster[14, 1],labels[14],fontsize=8)
plt.xlabel("Confirmed Cases")
plt.ylabel("Mean Temperature of the day")
plt.legend()
plt.show()


