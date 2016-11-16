#!/usr/bin/evn python

import os
import learning
import fileIO
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dir = "/Users/jonpdeaton/Documents/research/phamer_data/reference_features"
    file = "positive_features.csv"

    file = os.path.join(dir, file)

    ids, data = fileIO.read_feature_file(file, normalize=True)

    k_clusters = np.arange(10, 130, 10)
    k_clusters = np.apend(k_clusters, np.arange(100, 120))
    k_clusters = np.array(list(set(k_clusters)))
    sil_scores = [learning.silhouette_score(data, learning.kmeans(data, k, verbose=True)) for k in k_clusters]
    sil_scores = np.array(sil_scores)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(k_clusters, sil_scores, 'k-o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Mean Silhouette")
    ax.set_title("Phage k-mers")
    fig.savefig("kmeans_cluster.svg")



