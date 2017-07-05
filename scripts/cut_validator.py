#!/usr/bin/env python

import os
import argparse
import warnings
import logging
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt

import learning
import phamer
import cross_validate
import fileIO

# This is so that the PDF images created have editable text (For Adobe Illustrator)
matplotlib.rcParams['pdf.fonttype'] = 42

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class tester(object):

    def __init__(self):
        self.cut_directory = None
        self.output_directory = None
        self.all_features_files = None
        self.N_fold = 20
        self.validator = None
        self.cut_sizes = []
        self.cut_file_map = None
        self.aucs = None

    def get_cut_file_map(self):
        """
        This function makes a map from cutsize to a list of files with that cut size
        :param cuts_direcotry: A directory containing cut features files
        :return: A dictionary taht maps integers representing cut-sizes to lists of files
        """
        self.all_features_files = [os.path.join(self.cut_directory, file) for file in os.listdir(self.cut_directory) if file.endswith(".csv")]
        cut_map = {}
        for file in self.all_features_files:
            cut_size = get_cutsize_from_filename(file)
            if cut_size in cut_map:
                cut_map[cut_size] += [file]
            else:
                cut_map[cut_size] = [file]
        return cut_map

    def get_output_plot_filename(self):
        """
        This function generates a name for the output plot
        :return:
        """
        return os.path.join(self.output_directory, "cut_response.pdf")

    def test_cut_response(self):
        '''
        This function tests the effect of sequence cut length on Phamer learning algorithm
        :return: A dictionary mapping cut length to ROC AUC values
        '''
        self.cut_file_map = self.get_cut_file_map()
        self.cut_sizes = sorted(self.cut_file_map.keys())
        self.aucs = np.zeros(len(self.cut_sizes))
        if self.validator is None:
            self.validator = cross_validate.cross_validator()
            self.validator.scoring_function = phamer.score_points
        self.validator.N = self.N_fold
        i = 0
        for cut_size in self.cut_sizes:
            logger.info("Cross validating with cutsize: %d bp" % cut_size)
            phage_file = [file for file in self.cut_file_map[cut_size] if os.path.basename(file).startswith("phage")][0]
            bacteria_file = [file for file in self.cut_file_map[cut_size] if os.path.basename(file).startswith("bacteria")][0]
            self.validator.positive_data = fileIO.read_feature_file(phage_file, normalize=True, old=True)[1]
            self.validator.negative_data = fileIO.read_feature_file(bacteria_file, normalize=True, old=True)[1]
            phage_scores, bacteria_scores = self.validator.cross_validate()
            self.aucs[i] = learning.predictor_performance(phage_scores, bacteria_scores)[2]
            i += 1

        plt.figure(figsize=(9, 6))
        plt.plot(np.array([0] + self.cut_sizes) / 1000.0, [0] + list(self.aucs), 'b-o')
        plt.grid(True)
        plt.xlabel('Cut Size (kbp)')
        plt.ylabel('ROC AUC')
        plt.savefig(self.get_output_plot_filename())
        return dict(zip(self.cut_sizes, self.aucs))


def get_cutsize_from_filename(filename):
    """
    This function finds the cut size for a particular file
    for instance for the file: phage_kmer_count_k4_c100000_s0.csv, the cutsize is 100000
    :param filename: The name of the file
    :return: An integer specifying what the cutsize is
    """
    base = os.path.basename(filename)
    try:
        start = base.index("_c") + 2
        end = base.index('_', start)
        cut_size = int(base[start:end])
    except ValueError as e:
        logger.error("Could not parse filename: %s" % os.path.basename(filename))
        raise e
    return cut_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser("This script tests the cross validation performance impact of using cut sequences")
    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input_directory', required=True, help="Directory containing cuts")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_directory', required=True, help="Directory to put output files in")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-n', '--N_fold', default=20, type=int, help="N-fold cross validation")

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

    logger.info("Cross validating with cut sequence data")
    logger.info("Data directory: %s" % os.path.basename(args.input_directory))

    my_tester = tester()
    my_tester.N_fold = args.N_fold
    my_tester.cut_directory = args.input_directory
    my_tester.output_directory = args.output_directory
    my_tester.test_cut_response()