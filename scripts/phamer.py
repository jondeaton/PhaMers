#!/usr/bin/env python
"""
Phamer: Phage-finding algorithm that uses k-mer frequency comparison and t-SNE
Jonathan Deaton, Quake Lab, Stanford University, 2016
"""

import os
import argparse
import warnings
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# My stuff
import kmer
import learning
import basic
import fileIO

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class phamer_scorer(object):

    def __init__(self):

        self.input_directory = None
        self.features_file = None
        self.fasta_file = None

        self.data_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.positive_fasta = None
        self.negative_fasta = None
        self.positive_features_file = None
        self.negative_features_file = None
        self.find_data_files()

        self.output_directory = None

        self.data_ids = None
        self.data_points = None
        self.positive_ids = None
        self.positive_data = None
        self.negative_ids = None
        self.negative_data = None

        self.tsne_data = None

        self.length_requirement = 5000

        self.scoring_method = 'combo'
        self.all_scoring_methods = ['dbscan', 'kmeans', 'knn', 'svm', 'density', 'silhouette', 'combo']
        scoring_functions = [self.dbscan_score_points, self.kmeans_score_points, self.knn_score_points,
                             self.svm_score_points, self.density_score_points, self.silhouette_score_points,
                             self.combo_score_points]
        self.method_function_map = dict(zip(self.all_scoring_methods, scoring_functions))

        self.kmer_length = 4
        self.k_clusters = 86
        self.k_clusters_positive = 86
        self.k_clusters_negative = 20
        self.positive_bandwidth = 0.005
        self.negative_bandwidth = 0.01
        self.positive_eps = 2
        self.negative_eps = 2
        self.positive_min_samples = 2
        self.negative_min_samples = 2
        self.eps = [self.positive_eps, self.negative_eps]
        self.min_samples = [self.positive_min_samples, self.negative_min_samples]

        self.use_tsne_python = False
        self.tsne_perplexity = 30.0
        self.pca_preprocess = False
        self.pca_preprocess_red = 50
        self.tsne_figsize = (30, 24)

    def load_data(self):
        """
        This function loads the relevant data into a phamer_scorer object
        :param args: An argparse parsed arguments object from this script
        :return: None. scorer object is modified in place
        """
        # loading reference data
        if self.positive_features and os.path.exists(self.positive_features_file):
            logger.info("Reading positive features from: %s" % os.path.basename(self.positive_features_file))
            scorer.positive_ids, scorer.positive_data = fileIO.read_feature_file(self.positive_features_file, normalize=True)
        elif self.positive_fasta and os.path.exists(self.positive_fasta):
            logger.info("Counting positive k-mers from: %s" % os.path.basename(self.positive_fasta))
            scorer.positive_ids, scorer.positive_data = kmer.count(self.positive_fasta)

        if self.negative_features_file and os.path.exists(self.negative_features_file):
            logger.info("Reading negative features from: %s" % os.path.basename(self.negative_features_file))
            scorer.negative_ids, scorer.negative_data = fileIO.read_feature_file(self.negative_features_file, normalize=True)
        elif self.negative_fasta and os.path.exists(self.negative_fasta):
            logger.info("Counting negative k-mers from: %s" % os.path.basename(self.negative_fasta))
            scorer.negative_ids, scorer.negative_data = kmer.count(self.negative_fasta)

        if args.equalize_reference:
            scorer.equalize_reference_data()

        scorer.find_input_files()
        # Loading input data
        if os.path.exists(self.features_file):
            logger.info("Reading features from file: %s ..." % os.path.basename(self.features_file))
            scorer.data_ids, scorer.data_points = fileIO.read_feature_file(self.features_file)
        elif os.path.exists(self.fasta_file):
            logger.info("Calculating features of: %s" % os.path.basename(self.fasta))
            self.data_ids, self.data_points = kmer.count_file(self.fasta, self.kmer_length, normalize=False)
            self.features_file = "{base}_features.csv".format(base=os.path.splitext(self.fasta_file)[0])
            logger.info("Saving features to {file}...".format(file=self.features_file))
            fileIO.save_counts(self.data_points, self.data_ids, self.features_file)

        self.data_points = kmer.normalize_counts(self.data_points)

        if args.length_requirement:
            logger.info("Screening input by length: %d bp..." % args.length_requirement)
            scorer.screen_by_length()

    def screen_by_length(self, length_requirement=None):
        """
        This function screens input data by total length
        :param length_requirement: Minimum length requried for sequences
        :return: None
        """
        if length_requirement:
            self.length_requirement = length_requirement
        unknown_ids, unknown_sequences = fileIO.read_fasta(self.fasta_file)
        logger.debug("%d points before screening" % self.data_points.shape[0])
        long_ids = [unknown_ids[i] for i in xrange(len(unknown_ids)) if len(unknown_sequences[i]) >= self.length_requirement]
        self.data_points = self.data_points[np.in1d(self.data_ids, long_ids)]
        self.data_ids = np.array(long_ids)
        logger.debug("%d points after screening" % self.data_points.shape[0])

    def equalize_reference_data(self):
        """
        This function ensures that there are the same number of negative and positive data points
        :return: None
        """
        num_positive = self.positive_data.shape[0]
        num_negative = self.negative_data.shape[0]
        num_ref = min(num_positive, num_negative)
        logger.debug("Equalizing reference data to: %d data-points" % num_ref)
        self.positive_data[:, :num_ref]
        self.negative_data[:, :num_ref]
        self.positive_ids[:num_ref]
        self.negative_ids[:num_ref]
        self.num_positive = num_ref
        self.num_negative = num_ref

    def score_points(self):
        """
        This function scores a set of points against positive and negative data-points.
        :return: A list of scores as a numpy array
        """
        # Setting up some data
        self.num_points = self.data_points.shape[0]
        self.num_positive = self.positive_data.shape[0]
        self.num_negative = self.negative_data.shape[0]
        self.train = np.vstack((self.positive_data, self.negative_data))
        self.labels = np.append(np.ones(self.num_positive), np.zeros(self.num_negative))

        # decidign which scoring function to use
        scoring_function = self.method_function_map[self.scoring_method]

        # actually scoring points
        logger.debug("Scoring %d points. Method: %s ..." % (self.data_points.shape[0], self.scoring_method))
        self.scores = np.array(scoring_function())
        return self.scores

    # Scoring Algorithm Functions
    def proximity_metric(self, point, nearest_positive, nearest_negative):
        """
        This function scores a point for being near positive and away from negative
        :param point: A numpy array with the point to be scored
        :param nearest_positive: A numpy array containing the centroid of the nearest positive cluster
        :param nearest_negative: A numpy array containing the centroid of the nearest negative cluster
        :return: A score for that point. A higher score is if the point is near the positive and far from the negative
        """
        error_neg = np.linalg.norm(point - nearest_negative)
        error_pos = np.linalg.norm(point - nearest_positive)
        x = (error_neg - error_pos) / (error_pos + error_neg)
        x = np.tanh(x)
        return x

    def dbscan_score_points(self):
        """
        Scoring function for the dbscan method
        :return: A list of scores corresponding to the points
        """
        positive_assignment = learning.dbscan(self.positive_data, self.eps[0], self.min_samples[0])
        negative_assignment = learning.dbscan(self.negative_data, self.eps[1], self.min_samples[1])

        # resort to k-means if things go poorly...
        if max(positive_assignment) < 2:
            logger.warning("Clustering positive with k-means instead...")
            positive_assignment = learning.kmeans(self.positive_data, self.k_clusters_positive)
        if max(negative_assignment) < 2:
            logger.warning("Clustering negative with k-means instead...")
            negative_assignment = learning.kmeans(self.negative_data, self.k_clusters_negative)

        positive_centroids = learning.get_centroids(self.positive_data, positive_assignment)
        negative_centroids = learning.get_centroids(self.negative_data, negative_assignment)

        scores = [0] * self.num_points
        for i in xrange(self.num_points):
            point = self.data_points[i]
            closest_positive = learning.closest_to(point, positive_centroids)
            closest_negative = learning.closest_to(point, negative_centroids)
            scores[i] = [self.proximity_metric(point, closest_positive, closest_negative)]

        return scores

    def kmeans_score_points(self):
        """
        Scoring function for the kmeans method
        :return: A list of scores corresponding to the points
        """
        positive_assignment = learning.kmeans(self.positive_data, self.k_clusters)
        negative_assignment = learning.kmeans(self.negative_data, self.k_clusters)
        positive_centroids = learning.get_centroids(self.positive_data, positive_assignment)
        negative_centroids = learning.get_centroids(self.negative_data, negative_assignment)

        scores = np.zeros(self.num_points)
        for i in xrange(self.num_points):
            point = self.data_points[i]
            closest_positive = learning.closest_to(point, positive_centroids)
            closest_negative = learning.closest_to(point, negative_centroids)
            scores[i] = self.proximity_metric(point, closest_positive, closest_negative)
        return scores

    def svm_score_points(self):
        """
        Scoring function for the k-means method
        :return: A list of scores corresponding to the points
        """
        machine = learning.svm.NuSVC()
        machine.fit(self.train, self.labels)
        scores = machine.predict(self.data_points)
        return np.array(scores)

    def knn_score_points(self):
        """
        Scoring function for the knn method
        :return: A list of scores corresponding to the points
        """
        return 2 * learning.knn(self.data_points, self.train, self.labels, k=3)

    def density_score_points(self):
        """
        Scoring function for the density method
        :return: A list of scores corresponding to the points
        """
        scores = np.zeros(self.num_points)
        for i in xrange(self.num_points):
            point = self.data_points[i]
            pos_density = learning.get_density(point, self.positive_data, bandwidth=self.positive_bandwidth)
            neg_density = learning.get_density(point, self.negative_data, bandwidth=self.negative_bandwidth)
            score = pos_density - neg_density
            scores[i] = score
        return scores

    def silhouette_score_points(self):
        """
        Scoring function for the silhouette method
        :return: A list of scores corresponding to the points
        """
        positive_appended = np.append(self.positive_data, self.data_points, axis=0)
        negative_appended = np.append(self.positive_data, self.data_points, axis=0)
        positive_assignment = learning.kmeans(positive_appended, 86)
        negative_assignment = learning.kmeans(negative_appended, 86)
        pos_sils = learning.silhouettes(positive_appended, positive_assignment)
        neg_sils = learning.silhouettes(negative_appended, negative_assignment)
        scores = np.array(pos_sils[-self.num_points:] - neg_sils[-self.num_points:])
        return scores

    def combo_score_points(self):
        """
        Scoring function for the combo method
        :return: A list of scores corresponding to the points
        """
        self.scoring_method = 'knn'
        knn_scores = self.score_points()
        self.scoring_method = 'kmeans'
        cluster_scores = self.score_points()
        self.scoring_method = 'combo'
        return np.array(knn_scores) + np.array(cluster_scores)

    # Making output files
    def make_summary_file(self, args=None):
        """
        This function is for saving phamer scores to file
        :param args: An argparse parserd arguments
        :return: None
        """
        self.phamer_output_filename = self.get_phamer_output_filename()
        fileIO.save_phamer_scores(self.data_ids, self.scores, self.phamer_output_filename, args=args)

    def save_tsne_data(self, args=None):
        """
        This function saves t-SNE data
        :param args: An argparse parsed arguments object. Used to make a header summary
        :return: None
        """
        tsne_file = self.get_tsne_output_filename()
        ids = np.concatenate((self.data_ids, self.positive_ids, self.negative_ids))
        chops = (len(self.data_ids), len(self.positive_ids), len(self.negative_ids))
        fileIO.save_tsne_data(tsne_file, self.tsne_data, ids, args=args, chops=chops)

    # t-SNE
    def do_tsne(self):
        """
        This function does t-SNE on the positive, negative, and unknown data provided
        :return: None
        """
        all_data = np.vstack((self.data_points, self.positive_data, self.negative_data))
        # This is to save memory and potentially prevent memory error
        del self.data_points
        del self.positive_data
        del self.negative_data
        if self.use_tsne_python:
            try:
                # Try importing and using the tsne_python implementation...
                import tsne
                self.tsne_data = tsne.tsne(all_data, 2, self.tsne_perplexity)
            except:
                pass
        elif self.pca_preprocess:
            # This is to reduce memory requirement
            logger.info("Pre-processing with PCA...")
            pca_data = PCA(n_components=self.pca_preprocess_red).fit_transform(all_data)
            self.tsne_data = TSNE(perplexity=self.tsne_perplexity, verbose=True).fit_transform(pca_data)
        else:
            self.tsne_data = TSNE(perplexity=self.tsne_perplexity, verbose=True).fit_transform(all_data)
        logger.info("t-SNE complete.")

        logger.debug("Rearranging data...")
        chops = (self.num_points, self.num_positive, self.num_negative)
        self.data_points, self.positive_data, self.negative_data = basic.chop(all_data, chops)
        del all_data

    def make_tsne_plot(self, tsne_file=None):
        """
        This function makes a t-SNE plot of all points
        :param positive: The 2D t-SNE positive (phage) points to plot in a numpy array
        :param negative: The 2D t-SNE negative points to plot in a numpy array
        :param unknown: The 2D t-SNE unknown points to plot in a numpy array
        :return: None
        """
        tsne_data_points, tsne_positive, tsne_negative = basic.chop(self.tsne_data, [self.num_unknown, self.num_positive, self.num_negative])
        if tsne_file and os.path.isfile(tsne_file):
            logger.debug("Using t-SNE data from: %s" % os.path.basename(tsne_file))
            try:
                self.tsne_data = fileIO.read_tsne_file(tsne_file)
            except:
                logger.error("Failed to read t-SNE file: %s" % os.path.basename(tsne_file))

        if self.tsne_data is None and os.path.isfile(self.get_tsne_output_filename()):
            try_tsne_file = self.get_tsne_output_filename()
            logger.debug("self.tsne_data is None. Looking for t-SNE data in: %s" % os.path.basename(try_tsne_file))
            try:
                self.tsne_data = fileIO.read_tsne_file(try_tsne_file)
            except:
                logger.error("Failed to read t-SNE file: %s" % os.path.basename(try_tsne_file))

        if self.tsne_data is None:
            logger.warning("Did not have nor could retrieve t-SNE data. t-SNE plot was not made.")
        else:
            plt.figure(figsize=self.tsne_figsize)
            pos = plt.scatter(tsne_positive[:, 0], tsne_positive[:, 1], c=[0, 0, 1], label='positive', alpha=0.9, marker='o')
            neg = plt.scatter(tsne_negative[:, 0], tsne_negative[:, 1], c=[0, 0, 0], label='negative', alpha=0.9, marker='o')
            unk = plt.scatter(tsne_data_points[:, 0], tsne_data_points[:, 1], c=[1, 0, 0], label='unknown', alpha=0.9, marker='o')
            plt.legend(handles=[pos, neg, unk])
            plt.title('t-SNE output')
            plt.grid(True)
            file_name = self.get_plot_output_filename()
            plt.savefig(file_name)

    # Functions for finding files
    def find_data_files(self):
        """
        This function finds data files in their proper location from the data directory
        :return: None
        """
        # These are file locations in their default places
        self.positive_fasta = os.path.join(self.data_directory, "all_phage_genomes.fasta")
        self.negative_fasta = os.path.join(self.data_directory, "bacteria_genomes_2")
        self.positive_features_file = os.path.join(self.data_directory, "reference_features", "positive_features.csv")
        self.negative_features_file = os.path.join(self.data_directory, "reference_features", "negative_features.csv")

    def find_input_files(self):
        """
        This function finds input files if the input directory is properly set up
        :return: None
        """
        if self.input_directory and os.path.isdir(self.input_directory):
            fasta_files = [file for file in os.listdir(self.input_directory) if file.endswith('.fasta') or file.endswith('.fa')]
            if len(fasta_files) == 1:
                self.fasta_file = os.path.join(self.input_directory, fasta_files[0])

            features_files = [file for file in os.listdir(self.input_directory) if file.endswith('.csv')]
            if len(features_files) == 1:
                self.features_file = os.path.join(self.input_directory, features_files[0])

    # Functions for generating paths to output files
    def get_phamer_output_filename(self):

        return os.path.join(self.output_directory, "phamer_scores.csv")

    def get_tsne_output_filename(self):

        return os.path.join(self.output_directory, "tsne_coordinates.csv")

    def get_plot_output_filename(self):

        return os.path.join(self.output_directory, "tsne_plot_phamer.svg")

def score_points(scoring_data, positive_training_data, negative_training_data, method=None):
    """
    This is a function that will score data in the same way that the phamer_scoring object would but this
    works as a function with arguments rathern than as the method of an object. This is used in the cross validation
    script so that testing can be done with the same exact method of
    :param scoring_data: Data to score in a numpy array (rows = items)
    :param positive_training_data: Positive data in a numpy array
    :param negative_training_data: Negative data in a numpy array
    :return: A list of scores as a numpy array
    """
    scorer = phamer_scorer()
    if method:
        scorer.scoring_method = method
    scorer.data_points = scoring_data
    scorer.positive_data = positive_training_data
    scorer.negative_data = negative_training_data
    return scorer.score_points()

def decide_files(scorer, args):
    """
    This function decides which data files to use, based on those which were provided by the user
    :param scorer: Phamer scorer object
    :param args: Argparse parsed arguments object
    :return: None
    """

    if args.kmer_length:
        # This isn't a file but still needs to be loaded in if specified
        scorer.kmer_length = args.kmer_length

    if args.input_directory:
        scorer.input_directory = args.input_directory
        scorer.find_input_files()

    if args.data_directory:
        scorer.data_directory = args.data_directory
        scorer.find_data_files()

    # Deciding where the output directory should be
    if args.output_directory:
        # If a directory was provided by the use
        scorer.output_directory = args.output_directory
    else:
        # No explicit directory, try to make put it with the inputs
        if not scorer.input_directory and args.input_directory:
            scorer.input_drectory = args.input_directory
        elif not scorer.fasta_file and args.input_fasta:
            scorer.input_drectory = os.path.dirname(args.fasta_file)
        elif not scorer.features_file and args.features_file:
            scorer.input_drectory = os.path.dirname(args.features_file)
        scorer.output_directory = os.path.join(scorer.input_directory, "Phamer_output", "phamer")

    scorer.fasta_file = basic.decide_file(args.fasta_file, scorer.fasta_file)
    scorer.features_file = basic.decide_file(args.features_file, scorer.features_file)
    scorer.positive_fasta = basic.decide_file(args.positive_fasta, scorer.positive_fasta)
    scorer.negative_fasta = basic.decide_file(args.negative_fasta, scorer.negative_fasta)
    scorer.positive_features = basic.decide_file(args.positive_features, scorer.positive_features_file)
    scorer.negative_features = basic.decide_file(args.negative_features, scorer.negative_features_file)


if __name__ == '__main__':

    script_description = 'This script scores contigs based on feature similarity'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input_directory', help='Directory containing input files')
    input_group.add_argument('-fasta', '--fasta_file', help='Fasta compilation file of unknown sequences')
    input_group.add_argument('-features', '--features_file', help="Input feature file")
    input_group.add_argument('-tsne', '--tsne_file', help='Preprocessed t-SNE data file in csv format')

    data_group = parser.add_argument_group("Data")
    data_group.add_argument('-data', "--data_directory", help="Directory containing all relevant data files")
    data_group.add_argument('-p', '--positive_fasta', help='Fasta compilation file of positive sequences')
    data_group.add_argument('-n', '--negative_fasta', help='Fasta compilation or directory of negative seuquences')
    data_group.add_argument('-id', '--file_identifier', default='.fna', help='File identifier for fasta files in negative directory')
    data_group.add_argument('-pf', '--positive_features', help='positive feature file')
    data_group.add_argument('-nf', '--negative_features', help='negative feature file')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_directory', help='Directory to put output files')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-k', '--kmer_length', type=int, default=4, help='Length of k-mers analyzed')
    options_group.add_argument('-l', '--length_requirement', type=int, default=5000, help='Sequence length requirement')
    options_group.add_argument('-equal', '--equalize_reference', action='store_true', help="Use same number of reference")

    tsne_options_group = parser.add_argument_group("t-SNE Options")
    tsne_options_group.add_argument('-do_tsne', '--do_tsne', action='store_true', help='Flag to perform new t-SNE')
    tsne_options_group.add_argument('-pxty', '--perplexity', type=float, default=30, help='t-SNE Perplexity')
    tsne_options_group.add_argument('-plot', '--plot_tsne', action='store_true', help='Flag makes t-SNE plots')

    learning_options_group = parser.add_argument_group("Learning Options")
    learning_options_group.add_argument('-m', '--method', default='combo', help='Learning algorithm name')
    learning_options_group.add_argument('-eps', '--eps', type=float, default=2.1, help='DBSCAN eps parameter')
    learning_options_group.add_argument('-mp', '--minPts', type=int, default=2, help='DBSCAN minimum points per cluster parameter')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')

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

    # Making a new scoring object
    scorer = phamer_scorer()
    # Deciding between default files, and those provided by the use
    decide_files(scorer, args)
    logger.info("Loading data into scoring object...")
    scorer.load_data()

    if not os.path.isdir(scorer.output_directory):
        os.mkdir(scorer.output_directory)

    logger.info("Scoring points...")
    scorer.scores = scorer.score_points()
    logger.info("Writing scores to file...")
    scorer.make_summary_file(args=args)

    # t-SNE
    if args.do_tsne:
        logger.info("Performing t-SNE...")
        if args.perplexity:
            scorer.tsne_perplexity = args.perplexity
        scorer.do_tsne()
        logger.info("Saving t-SNE data...")
        scorer.save_tsne_data(args=args)

    # Plotting
    if args.plot_tsne:
        logger.info("Creating t-SNE plot...")
        scorer.make_tsne_plot()
        logger.info("Done creating t-SNE plot.")

    logger.info("Phamer complete.")
