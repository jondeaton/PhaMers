#!/usr/bin/env python
"""
cross_validate.py

This script is for doing N-fold cross validation of the Phamer scoring algorithm
"""

import os
import argparse
import fileIO
import learning
import kmer
import phamer
import warnings
import logging
import numpy as np
import matplotlib
warnings.simplefilter('ignore', UserWarning)
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class cross_validator(object):

    def __init__(self):

        self.positive_ids = None
        self.negative_ids = None
        self.positive_data = None
        self.negative_data = None
        self.positive_scores = None
        self.negative_scores = None
        self.equalize_reference = False

        self.N = 20
        self.method = None
        self.scoring_function = None
        self.score_threshold = 0
        self.roc_figsize = (9, 7)
        self.output_directory = "cross_validation"

    def cross_validate(self):
        """
        This function is for cross validation of a scoring metric
        :return: The scores for the gold standard positive and negative points
        """
        self.num_positive = self.positive_data.shape[0]
        self.num_negative = self.negative_data.shape[0]
        if self.equalize_reference and self.num_positive != self.num_negative:
            num_ref = min(self.num_positive, self.num_negative)
            logger.debug("Equalizing reference data to: %d data-points" % num_ref)
            self.positive_data = self.positive_data[:num_ref]
            self.negative_data = self.negative_data[:num_ref]
            self.positive_ids = self.positive_ids[:num_ref]
            self.negative_ids = self.negative_ids[:num_ref]
            self.num_positive = num_ref
            self.num_negative = num_ref

        positive_asmt = np.arange(self.num_positive) % self.N
        negative_asmt = np.arange(self.num_negative) % self.N

        np.random.shuffle(positive_asmt)
        np.random.shuffle(negative_asmt)

        self.positive_scores = np.zeros(self.num_positive)
        self.negative_scores = np.zeros(self.num_negative)

        for n in xrange(self.N):
            logger.info('Iteration %d/%d' % (1 + n, self.N))
            # Data segmentation, and sub-selection
            where_positive = (positive_asmt == n)
            where_negative = (negative_asmt == n)
            positive_sub_div_size = np.sum(where_positive)

            scoring_data = np.vstack((self.positive_data[where_positive], self.negative_data[where_negative]))
            pos_training_data = self.positive_data[np.invert(where_positive)]
            neg_training_data = self.negative_data[np.invert(where_negative)]

            # Sub-selection scoring
            scores = self.scoring_function(scoring_data, pos_training_data, neg_training_data, method=self.method)

            self.positive_scores[where_positive] = scores[:positive_sub_div_size]
            self.negative_scores[where_negative] = scores[positive_sub_div_size:]

        logger.info("%d-fold cross validation complete." % self.N)
        return self.positive_scores, self.negative_scores

    def plot_score_distributions(self):
        """
        This function makes nice looking distribution plots of postiive and negative_data scores
        :param positive_scores: The scores for the positive data
        :param negative_scores: The scores for the negative data
        :param file_name: The filename to save the plot to
        :return: None
        """
        fig, ax = plt.subplots()
        ll = min([min(self.negative_scores), min(self.positive_scores)])
        ul = max([max(self.negative_scores), max(self.positive_scores)])
        logger.debug("x limits: %.2f, %.2f" % (ll, ul))
        ranqe = ul - ll
        x = np.linspace(ll - 0.2 * ranqe, ul + 0.2 * ranqe, 1000)
        logger.debug("Performing KDE on scores...")

        pdf_positive = stats.gaussian_kde(self.positive_scores)
        pdf_negative = stats.gaussian_kde(self.negative_scores)
        y_pos = pdf_positive(x)
        y_neg = pdf_negative(x)

        logger.debug("Filling curves...")
        ax.fill_between(x, y_pos, facecolor='blue', alpha=0.4, label='%d positive' % len(self.positive_scores))
        ax.fill_between(x, y_neg, facecolor='red', alpha=0.4, label='%d negative' % len(self.negative_scores))
        plt.legend(loc='best')
        plt.grid(True)
        file_name = self.get_score_distribution_filename()
        if file_name and not os.path.isdir(os.path.dirname(file_name)):
            os.mkdir(os.path.dirname(file_name))
        plt.savefig(file_name)
        plt.close()

    def plot_ROC(self):
        """
        This function makes a receiver operator characteristic curve plot
        :return: None
        """
        fpr, tpr, roc_auc = learning.predictor_performance(self.positive_scores, self.negative_scores)
        plt.figure(figsize=self.roc_figsize)
        plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(b=True, which='minor')
        file_name = self.get_roc_curve_filename()
        plt.savefig(file_name)
        plt.close()

    def make_metrics_file(self):
        """
        This function makes a summary file with all the information about the cross validation run
        :return: None
        """
        summary_text = "# Cross Validation Prediction Metrics"
        file_name = self.get_summary_filename()
        with open(file_name, 'w') as f:
            f.write(summary_text + "\n")
            f.close()
        metrics_series = learning.get_predictor_metrics(self.positive_scores, self.negative_scores, threshold=self.score_threshold)
        metrics = ['tp', 'fp', 'fn', 'tn', 'tpr', 'fpr', 'fnr', 'tnr', 'ppv', 'npv', 'fdr', 'acc']
        metric_names = ["True Positives", "False Positives", "False Negatives", "True Negatives",
                        "True Positive Rate", "False Positive Rate", "False Negative Rate", "True Negative Rate"
                        "Positive Predictive Value", "Negative Predictive Value", "False Discovery Rate",
                        "Accuracy"]
        file_name = self.get_metric_filename()
        metrics_series.to_csv(file_name, sep="\t", mode='a')

    def make_summary_file(self, id_label_map=None):
        """
        This function makes a file that contains each positive id and it's associated score, sorted
        :return: None
        """
        text = "# Cross Validation Scores"
        sorted_positive_scores = sorted(self.positive_scores)
        sorted_ids = [id for (score, id) in sorted(zip(self.positive_scores, self.positive_ids))]
        for i in xrange(len(sorted_ids)):
            id = sorted_ids[i]
            score = sorted_positive_scores[i]
            if id_label_map is None:
                text += "\n{id}\t{score}".format(id=id, score=score)
            else:
                text += "\n{id}\t{score}\t{label}".format(id=id, score=score, label=id_label_map[id])
        file_name = self.get_summary_filename()
        f = open(file_name, 'w')
        f.write(text)
        f.close()

    def cross_validate_all_algorithms(self):
        """
        For cross validation of several different learning algorithms on a particular set of data
        :return: None
        """
        plt.figure(figsize=self.roc_figsize)
        for method in self.methods:
            self.method = method
            logger.info("%d-Fold Cross Validation" % self.N)
            logger.info("Algorithm: %s" % self.method.upper())
            positive_scores, negative_scores = self.cross_validate()
            fpr, tpr, roc_auc = learning.predictor_performance(positive_scores, negative_scores)
            plt.plot(fpr, tpr, label='%s - AUC = %0.3f' % (method.upper(), roc_auc))

        logger.info("Making ROC plot for all algorithms...")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", title="Algorithm")
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(b=True, which='minor')

        file_name = self.get_combined_roc_curve_filename()
        plt.savefig(file_name)
        plt.close()
        logger.info("Saved ROC plot...")

    # filename makers
    def get_score_distribution_filename(self):
        return os.path.join(self.output_directory, 'score_distributions.svg')

    def get_roc_curve_filename(self):
        return os.path.join(self.output_directory, "roc.svg")

    def get_combined_roc_curve_filename(self):
        return os.path.join(self.output_directory, "all_algorithms_roc.svg")

    def get_metric_filename(self):
        return os.path.join(self.output_directory, "metrics.txt")

    def get_summary_filename(self):
        return os.path.join(self.output_directory, "scores.txt")

if __name__ == '__main__':

    script_description = "This script is for doing N-fold cross validation of the Phamer scoring algorithm"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-pf', '--positive_features_file', help="Positive features file")
    input_group.add_argument('-nf', '--negative_features_file', help="Negative features file")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_directory', help="Output directory for plots")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-N', '--N_fold', default=20, type=int, help="Number of iteration in N-fold cross validation")
    options_group.add_argument('-m', '--method', default='combo', help="Scoring algorithm method")
    options_group.add_argument('-a', '--test_all', action='store_true', help="Flag to cross validate all algorithms")
    options_group.add_argument('-l', '--labels_file', help="Label file mapping id to label")
    options_group.add_argument('-equal', '--equalize_reference', action='store_true', help="Use same number of reference data from each")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help="Verbose output")
    console_options_group.add_argument('--debug', action='store_true', default=False, help="Debug console")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')

    validator = cross_validator()
    validator.scoring_function = phamer.score_points
    validator.method = args.method
    validator.N = args.N_fold
    validator.method = args.method
    validator.output_directory = args.output_directory

    positive_ids, positive_data = fileIO.read_feature_file(args.positive_features_file)
    negative_ids, negative_data = fileIO.read_feature_file(args.negative_features_file)

    validator.positive_ids = positive_ids
    validator.negative_ids = negative_ids
    validator.positive_data = kmer.normalize_counts(positive_data)
    validator.negative_data = kmer.normalize_counts(negative_data)
    validator.equalize_reference = args.equalize_reference

    validator.cross_validate()

    logger.info("Plotting score distributions...")
    validator.plot_score_distributions()
    logger.info("Plotting ROC curve...")
    validator.plot_ROC()
    logger.info("Making metrics file...")
    validator.make_metrics_file()
    logger.info("Making summary file...")
    id_lineage_map = fileIO.read_label_file(args.labels_file)
    validator.make_summary_file(id_label_map=id_lineage_map)

    if args.test_all:
        logger.info("Validating all algorithms...")
        validator.methods = ['dbscan', 'kmeans', 'knn', 'svm', 'density', 'combo']
        validator.cross_validate_all_algorithms()

    logger.info("Cross validation complete.")