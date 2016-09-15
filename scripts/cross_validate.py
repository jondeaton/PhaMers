#!/usr/bin/env python
'''
cross_validate.py
This script is for doing N-fold cross validation of the Phamer scoring algorithm
'''

import os
import sys
import time
import argparse
import kmer
import phamer
import warnings
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
warnings.simplefilter('ignore', UserWarning)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyqt_fit import kde

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def make_distributions(positive_scores, negative_scores, file_name='distributions.svg'):
    '''
    This function makes nice looking distribution plots of postiive and negative_data scores
    :param positive_scores: The scores for the positive data
    :param negative_scores: The scores for the negative data
    :param file_name: The filename to save the plot to
    :return: None
    '''
    fig, ax = plt.subplots()
    ll = min([min(negative_scores), min(positive_scores)])
    ul = max([max(negative_scores), max(positive_scores)])
    rng = ul - ll
    xs = np.linspace(ll - 0.5 * rng, ul + 0.5 * rng, 1000)
    est1 = kde.KDE1D(positive_scores[:])
    est2 = kde.KDE1D(negative_scores[:])
    ax.fill(xs, est1(xs), fc='blue', alpha=0.4, label='%d Phage (bw=%.3g)' % (len(positive_scores), est1.bandwidth))
    ax.fill(xs, est2(xs), fc='red', alpha=0.4, label='%d Bacteria (bw=%.3g)' % (len(negative_scores), est2.bandwidth))
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(file_name)


def make_ROC(positive_scores, negative_scores, file_name='xvalidation_ROC.svg'):
    '''
    This function makes a nice looking ROC curve
    :param positive_scores: The positive scores
    :param negative_scores: The negative scores
    :param roc_filename: The name of the file to save the image to
    :return: None
    '''
    fpr, tpr, roc_auc = predictor_performance(positive_scores, negative_scores)
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='minor')


def predictor_performance(positive_scores, negative_scores):
    '''
    This function calculates the performance of a prediction algorithm
    :param positive_scores: The scores for gold standard positive datapoints
    :param negative_scores: The scores for gold standard negative datapoints
    :return: A tuple containing false positive rate, true positive rate, and ROC area under the curve.
    '''
    truth = np.append(np.ones(len(positive_scores)), np.zeros(len(negative_scores))).astype(bool)
    predictions = np.append(positive_scores, negative_scores)
    fpr, tpr, _ = roc_curve(truth, predictions)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def cross_validate(positive_data, negative_data, N, method='combo', eps=[0.01, 0.01], min_pts=[2, 2]):
    '''
    This function is for cross validation of the Phamer learnign algorithm
    :param positive_data: The gold standard positive poitns
    :param negative_data: The gold standard negative points
    :param N: Number of iterations of cross validation
    :param eps: DBSCAN distance threshold
    :param min_pts: DBSCAN minimum points per cluster
    :return: The scores for the gold standard positive and negative points
    '''
    positive_asmt = np.arange(positive_data.shape[0]) % N
    negative_asmt = np.arange(negative_data.shape[0]) % N

    np.random.shuffle(positive_asmt)
    np.random.shuffle(negative_asmt)

    positive_scores, negative_scores = np.array([]), np.array([])

    tic = time.time()

    for n in xrange(N):
        logger.info('Iteration %d of %d' % (1 + n, N))
        which_positive = positive_asmt == n
        which_negative = negative_asmt == n

        data = np.vstack((positive_data[which_positive], negative_data[which_negative]))
        scores = phamer.score_points(data, positive_data[np.invert(which_positive)], negative_data[np.invert(which_negative)], method=method, eps=eps, min_samples=min_pts)

        positive_scores = np.append(positive_scores, scores[:np.sum(which_positive)])
        negative_scores = np.append(negative_scores, scores[np.sum(which_positive):])

    timed = time.time() - tic
    logger.info("N-Fold cross validation done. %dh %dm %ds" % (timed // 3600, (timed % 3600) // 60, timed % 60))
    return positive_scores, negative_scores


def test_cut_response(directory, N=5, eps=[0.1,0.1], min_pts=[2,2], file_name='cut_response.svg', method='combo'):
    '''
    This function tests the effect of sequence cut length on Phamer learning algorithm
    :param directory: The directory that contains the k-mer count data for different sized cute
    :param N: Number of iterations of DBSCAN
    :param eps: DBSCAN distance threshold
    :param min_pts: DBSCAN minimum points per cluster
    :param filename: The filename to save the output image to
    :return: A dictionary mapping cut length to ROC AUC values
    '''
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    cuts = [2000, 5000, 7500, 10000, 100000]
    aucs = []
    for cut in cuts:
        phage_file = [os.path.join(directory, file) for file in files if 'phage' in file and '_c%s_' % str(cut) in file][0]
        bacteria_file = [os.path.join(directory, file) for file in files if 'bacteria' in file and '_c%s_' % str(cut) in file][0]

        phage_kmers = kmer.read_kmer_file(phage_file, normalize=True, old=True)[1]
        bacteria_kmers = kmer.read_kmer_file(bacteria_file, normalize=True, old=True)[1]

        logger.info("loaded k-mers from: %s and %s" % (os.path.basename(phage_file), os.path.basename(bacteria_file)))
        phage_scores, bacteria_scores = cross_validate(phage_kmers, bacteria_kmers, N, eps=eps, min_pts=min_pts, method=method)
        auc = predictor_performance(phage_scores, bacteria_scores)[2]
        aucs.append(auc)

    plt.figure(figsize=(9, 6))
    plt.plot(np.array([0] + cuts) / 1000.0, [0] + aucs, 'b-o')
    plt.grid(True)
    plt.xlabel('Cut Size (kbp)')
    plt.ylabel('ROC AUC')
    plt.savefig(file_name)
    return dict(zip(cuts, aucs))


def cross_validate_all(positive_data, negative_data, N,image_filename='combined_ROC.svg', output=''):
    '''
    For validation all learning algorithms
    :param positive_data: The gold standard positive poitns
    :param negative_data: The gold standard negative points
    :param N: Number of iterations of cross validation
    :return: None
    '''
    eps = [0.012, 0.012]
    min_pts = [2, 2]
    methods = ['dbscan', 'kmeans', 'knn', 'svm', 'density', 'combo']
    plt.figure(figsize=(9, 7))

    for method in methods:
        logger.info("%d-Fold Cross Validation on: %s" % (N, method.upper()))
        positive_scores, negative_scores = cross_validate(positive_data, negative_data, N, method=method)
        fpr, tpr, roc_auc = predictor_performance(positive_scores, negative_scores)
        plt.plot(fpr, tpr, label='%s - AUC = %0.3f' % (method.upper(), roc_auc))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", title="Learning Algorithm")
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='minor')
    plt.savefig(image_filename)


if __name__ == '__main__':

    script_description = "This script is for doing N-fold cross validation of the Phamer scoring algorithm"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-p', '--positive', type=str, help='Positive k-mer counts')
    input_group.add_argument('-n', '--negative', type=str, help='Negative k-mer counts')
    input_group.add_argument('-cuts', '--cuts', type=str, help='analyse cut response from directory')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output', type=str, default='', help='Output directory for plots')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-N', '--N_fold', default=5, type=int, help="Number of iteration in N-fold cross validation")
    options_group.add_argument('-m', '--method', default='combo', type=str, help='Scoring algorithm method')
    options_group.add_argument('-a', '--CV_all', action='store_true', help='Flag to cross validate all algorithms')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')
    args = parser.parse_args()

    positive_file = args.positive
    negative_file = args.negative
    N = args.N_fold
    method = args.method
    all = args.CV_all
    cuts = args.cuts
    output = args.output

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')


    positive_data = kmer.read_kmer_file(positive_file)[1]
    negative_data = kmer.read_kmer_file(negative_file)[1]

    positive_data = kmer.normalize_counts(positive_data)
    negative_data = kmer.normalize_counts(negative_data)

    positive_scores, negative_scores = cross_validate(positive_data, negative_data, N, method=method)

    logger.info("Making plots...")
    distribution_file = os.path.join(output, 'distributions.svg')
    make_distributions(positive_scores, negative_scores, file_name=distribution_file)

    roc_filename = os.path.join(output, 'xvalidation_ROC.svg')
    make_ROC(positive_scores, negative_scores, file_name=roc_filename)

    if cuts:
        logger.info("Testing cut response... Directory: %s" % cuts)
        cut_filename = os.path.join(output, 'cut_response.svg')
        test_cut_response(cuts, N=N, method=method, file_name=cut_filename)

    if all:
        cross_validate_all(positive_data, negative_data, N, output=output)

    logger.info("Cross Validation Complete")