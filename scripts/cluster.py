#!/usr/bin/env python2.7
"""
cluster.py

This function is for plotting the average cluster silhouette as a function of
the number of clusters using for assignment
"""

import os
import learning
import fileIO
import numpy as np
import matplotlib

try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--features_file", required=True, help="Features file")
    parser.add_argument("-out", "--output_file", required=False, help="Filename for the output plot")
    args = parser.parse_args()

    ids, data = fileIO.read_feature_file(args.features_file, normalize=True)

    k_clusters = np.arange(10, 600, 10)
    k_clusters = np.array(sorted(list(set(k_clusters))))
    sil_scores = np.zeros(k_clusters.shape)
    sil_score_std = np.zeros(k_clusters.shape)

    num_repeats = 5

    for i in xrange(k_clusters.shape[0]):
        k = k_clusters[i]

        means = np.zeros(num_repeats)
        for j in xrange(num_repeats):
            sils = learning.silhouette_score(data, learning.kmeans(data, k, verbose=True))
            means[j] = np.mean(sils)

        sil_scores[i] = np.mean(means)
        sil_score_std[i] = np.std(means)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(k_clusters, sil_scores)
    ax.errorbar(k_clusters, sil_scores, yerr=sil_score_std)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette")
    ax.set_title(args.features_file)

    if args.output_file is None:
        filename = "%s_sil.svg" % os.path.splitext(os.path.basename(args.features_file))[0]
    else:
        filename = args.output_file
    fig.savefig(filename)
