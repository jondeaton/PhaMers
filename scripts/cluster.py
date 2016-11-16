#!/usr/bin/env python
"""
cluster.py

This function is for plotting the average cluster silhouette as a function of
the number of clusters using for assignment
"""

import os
import learning
import fileIO
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--features_file", required=True, help="Features file")
    parser.add_argument("-out", "--output_file", required=False, help="Filename for the output plot")
    args = parser.parse_args()

    ids, data = fileIO.read_feature_file(args.features_file, normalize=True)

    k_clusters = np.arange(10, 130, 10)
    k_clusters = np.append(k_clusters, np.arange(100, 120))
    k_clusters = np.array(list(set(k_clusters)))
    sil_scores = np.zeros(k_clusters.shape)
    sil_score_std = np.zeros(k_clusters.shape)

    for i in xrange(k_clusters.shape[0]):
        k = k_clusters[i]
        sils = learning.silhouette_score(data, learning.kmeans(data, k, verbose=True))
        sil_scores[i] = np.mean(sil_scores)
        sil_score_std[i] = np.std(sil_scores)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.errorbar(k_clusters, sil_scores, yerr=sil_score_std)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Mean Silhouette")
    ax.set_title("Phage k-mers")

    if args.output_file is None:
        filename = "%s_sil.svg" % os.path.splitext(os.path.basename(args.features_file))[0]
    else:
        filename = args.output_file
    fig.savefig(filename)



