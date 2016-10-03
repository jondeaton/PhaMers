#!/usr/bin/env python
"""
This script is for wrapper functions of machine learning algorithms
"""

from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
import logging
import pandas as pd

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def cluster_deviations(data, assignment):
    """
    A function for getting mean average distance (MAD) from the centroid of a number of clustered data-points
    :param data: The data-points in a 2 dimensional numpy array
    :param assignment: A clustering assignment array for each of the data-points
    :return: A numpy array containing the MAD for the points from each cluster
    """
    num_clusters = max(assignment)
    centroids = get_centroids(data, assignment)
    deviations = np.zeros(num_clusters)
    for cluster in xrange(num_clusters):
        which = [i for i in xrange(data.shape[0]) if assignment[i] == cluster]
        deviations[cluster] = np.mean(distances(centroids[cluster], data[which]))
    return deviations


def distances(vector, data):
    """
    A function for finding the distances from one point to many
    :param vector: A numpy array specifying the point in space to find distances from.
    :param data: A list of points in a numpy array to find distances to
    :return: A numpy array with each element being the distance from the vector to the data-points
    """
    if len(vector.shape) == 1:
        vector = np.array([vector])
    return np.linalg.norm(np.repeat(vector, data.shape[0], axis=0) - data, axis=1)


def closest_to(point, picks):
    """
    This function returns one point from many which is closest to another point
    :param point: A numpy array specifying the point of interest to compare proximity of all other points
    :param picks: A numpy array with rows specifying other points to pick the closest from
    :return: A numpy array which is the row of the "picks" array that is closest to the "point" vector
    """
    return picks[np.argmin(distances(point, picks))]



def get_centroids(data, assignment):
    """
    A function for getting the centroids of clustered data-points
    :param data: The data-points as rows in a 2 dimensional numpy array
    :param assignment: A numpy array containing the cluster assignments. Cluster indexing should start at zero.
    :return: A numpy array with rows being the centroid points for each cluster of data-points
    """
    num_clusters = 1 + max(assignment)

    if num_clusters == 0:
        logger.warning("No clusters assigned to data.")

    centroids = [np.mean(data[assignment == cluster_idx], axis=0) for cluster_idx in assignment if cluster_idx != -1]
    return np.array(centroids)


def silhouettes(data, assignment):
    """
    This function returns the silhouette values for data-points given a particular cluster assignment, which are values
    from -1 to 1 that describe how well the data-points are assigned to clusters
    :param data: A numpy array of row vectors specifying the data-points
    :param assignment: A list of integers specifying the cluster that each data-point was assigned to. -1 = not assigned
    :return: A numpy array of silhouette values corresponding to each data-point
    """
    return silhouette_samples(data, assignment)


def cluster_silhouettes(data, assignment, cluster):
    """
    This function computes the silhouette values for a single cluster
    :param data: A numpy array of row vectors specifying the data-points
    :param assignment: A list of integers specifying the cluster that each data-point was assigned to. -1 = not assigne
    :param cluster: The cluster to calculate silhouette values for
    :return: A numpy array with silhouette values for each point within the cluster
    """
    ss = silhouettes(data, assignment)
    return np.array([ss[i] for i in xrange(len(assignment)) if assignment[i] == cluster])


def get_density(point, data, bandwidth=0.1):
    """
    This function returns the density of the data at the given point, using t-distribution kernel density
    :param point: A numpy array vector specifying a point in space to evaluate the density at
    :param data: A 2D numpy array of points (rows)
    :return: A float representing the density of data-points at the given point
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    return kde.score_samples(np.array([point]))[0]


def knn(query, data, labels, k=3):
    """
    K-Nearest Neighbors wrapper method
    :param query: The point to search a label for as a numpy array
    :param data: The data to compare the query to as a numpy array where rows are points
    :param labels: The labels of each point in the data array
    :param k: Number of nearest neighbors to consider
    :return: The guessed classification
    """
    near_labels = labels[np.argsort(distances(query, data))[:k]]
    m = int(stats.mode(near_labels)[0])
    r = float(np.count_nonzero(near_labels == m)) / k
    return m


def kmeans(data, k):
    """
    K-Means clustering wrapper function
    :param data: The data to cluster as a numpy array with data-points being rows
    :param k: The number of clusters
    :return: A numpy array with elements corresponsing to the cluster assignment of each point
    """
    assignment = KMeans(n_clusters=k).fit(data).labels_
    ss = silhouette_score(data, assignment)
    logger.debug("K-means clustering (k=%d) silhouette score: %f" % (k, np.mean(ss)))
    return assignment


def dbscan(data, eps, min_samples):
    """
    DBSCAN wrapper function
    :param data: A numpy with rows that are data-points in a vector space
    :param eps: A float specifying the maximum distance that a point can be away from a cluster to be included
    :param min_samples: The minimum number of samples per cluster
    :return: An array specifying the cluster assignment of each data-point
    """
    asmt = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
    num_clusters = 1 + max(asmt)
    pct_unassigned = 100.0 * np.sum(asmt == -1) / float(len(asmt))
    logger.debug('%d positive clusters, %.1f%% unassigned' % (num_clusters, pct_unassigned))
    return asmt


def predictor_performance(positive_scores, negative_scores):
    """
    This function calculates the performance of a prediction algorithm
    :param positive_scores: The scores for gold standard positive data-points
    :param negative_scores: The scores for gold standard negative data-points
    :return: A tuple containing false positive rate, true positive rate, and ROC area under the curve.
    """
    truth = np.append(np.ones(len(positive_scores)), np.zeros(len(negative_scores))).astype(bool)
    predictions = np.append(positive_scores, negative_scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, predictions)
    roc_area = auc(false_positive_rate, true_positive_rate)
    return false_positive_rate, true_positive_rate, roc_area

def get_truth_table(positive_scores, negative_scores, threshold=0):
    """
    This function finds true positive rate, false positive rae, false negative rate, truth negative rate
    :param positive_scores: A numpy array of scores for positive data
    :param negative_scores: A numpy array of scores for negative data
    :return: A tuple containing TPR, FPR, FNR, TNR in that order
    """
    tp = np.sum(positive_scores >= threshold)
    fp = np.sum(negative_scores >= threshold)
    fn = np.sum(positive_scores < threshold)
    tn = np.sum(negative_scores < threshold)
    tpr = float(tp) / (tp + fn)
    fpr = float(fp) / (fp + tn)
    fnr = 1 - tpr
    tnr = 1 - fpr
    return tpr, fpr, fnr, tnr

def get_predictor_metrics(positive_scores, negative_scores, threshold=0):
    """
    This function calculates positive predictive value (PPV), negative predictive value, false discovery rate,
    and accuracy of a scoring metric
    :param positive_scores: A numpy array of scores for positive data
    :param negative_scores: A numpy array of scores for negative data
    :param threshold:
    :return: A pandas dataframe containing prediction metrics
    """
    metrics = ['tp', 'fp', 'fn', 'tn', 'tpr', 'fpr', 'fnr', 'tnr', 'ppv', 'npv', 'fdr', 'acc']
    series = pd.Series(index=metrics)
    series.tp = np.sum(positive_scores >= threshold)
    series.fp = np.sum(negative_scores >= threshold)
    series.fn = np.sum(positive_scores < threshold)
    series.tn = np.sum(negative_scores < threshold)
    series.tpr, series.fpr, series.fnr, series.tnr = get_truth_table(positive_scores, negative_scores, threshold=threshold)
    series.ppv = float(series.tp) / (series.tp + series.fp)
    series.npv = float(series.tn) / (series.tn + series.fn)
    series.fdr = 1 - series.ppv
    series.acc = float(series.tp + series.tn) / (series.tp + series.fp + series.fn + series.tn)
    return series
