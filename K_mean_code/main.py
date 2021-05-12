import os
from Utils import DataLoader
from Model import KMeans
import matplotlib.pyplot as plt
import numpy as np

mnist_data = DataLoader('K_mean_code/dataset')
tr_data, tr_class_labels, tr_subclass_labels = mnist_data.loaddata()

print(tr_data.shape)

mnist_data.plot_imgs(tr_data,25,True)

kmeans = KMeans(n_clusters=10,max_iter=200)
kmeans.fit(tr_data,tr_class_labels)
mnist_data.plot_imgs(kmeans.centroids, len(kmeans.centroids))
plt.plot(range(kmeans.iterations),kmeans.loss_per_iteration)
plt.show()

for key,data in list(kmeans.clusters['data'].items()):
    print('Cluster: ',key, 'Label:',kmeans.clusters_labels[key])
    mnist_data.plot_imgs(data[:min(25,data.shape[0])],min(25,data.shape[0]))

print('[cluster_label,no_occurence_of_label,total_samples_in_cluster,cluster_accuracy]\n',kmeans.clusters_info)
print('Accuracy:',kmeans.accuracy)