#!/usr/bin/env python
'''
This script is for transforming an array of k-mer count vectors
'''

__version__ = 1.0
__author__ = "Jonathan Deaton (jonpauldeaton@gmail.com)"
__license__ = "None"

import argparse
import fileIO
import logging
import numpy as np

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DNA = 'ATGC'

def get_transformed_indicies(k, num_symbols, reposition_indicies=None, index_substitution_map=None):
    """
    This function transforms things...
    :param k:
    :return:
    """
    b = num_symbols
    # This line takes an index j like (from 'CTG' -> 30) and writes it as [3, 2, 1]
    decompose = lambda j: np.array([((j % (b ** (i + 1))) - (j % (b ** i))) / (b ** i) for i in xrange(k, -1, -1)])
    if reposition_indicies is None:
        reposition_indicies = np.arange(k)
    if index_substitution_map is None:
        index_substitution_map = {i:i for i in xrange(num_symbols)}
    substitute = lambda indicies: [index_substitution_map[i] for i in indicies]
    # This line makes magnitudes [b^2, b^1, b^0] to dot
    magnitude_multipliers = np.array([k ** i for i in xrange(k)])
    # This function transforms an index
    trans = lambda j: np.dot(substitute(decompose(j)[reposition_indicies]), magnitude_multipliers)
    # This is all of the transformed indicies
    return np.array([trans(j) for j in np.arange(num_symbols ** k)])


def get_reverse_complement_indicies(k):
    reverse_reposition_indicies = np.arange(k - 1, -1, -1)
    DNA = 'ATGC'
    DNA_comlement_index_substitution_map = {0: 1, 1: 0, 2: 3, 3: 2}
    return get_transformed_indicies(k, len(DNA), reposition_indicies=reverse_reposition_indicies, index_substitution_map=DNA_comlement_index_substitution_map)


def get_reverse_indicies(k, symbols=DNA):
    reverse_reposition_indicies = np.arange(k - 1, -1, -1)
    return get_transformed_indicies(k, len(symbols), reposition_indicies=reverse_reposition_indicies)


def get_DNA_complement_indicies(k):
    DNA = 'ATGC'
    DNA_comlement_index_substitution_map = {0: 1, 1: 0, 2: 3, 3: 2}
    return get_transformed_indicies(k, len(DNA), index_substitution_map=DNA_comlement_index_substitution_map)


def transform_kmers(counts, reverse=True, complement=False, symbols=DNA):
    """

    :param counts:
    :param reverse:
    :param complement:
    :param symbols:
    :return:
    """
    num_symbols = len(symbols)
    k = int(round(np.math.log(counts.shape[1], num_symbols)))
    counts = counts.transpose()
    if reverse and not complement:
        counts += counts[get_reverse_indicies(k)]
    if complement and not reverse:
        counts += counts[get_DNA_complement_indicies(k)]
    if reverse and complement:
        counts += counts[get_reverse_complement_indicies(k)]
    return counts.transpose()


if __name__ == '__main__':

    script_description = 'This script transforms k-mer vectors'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=True, help='Input k-mer count (features) file')

    output_group = parser.add_argument_group("Outputs")
    parser.add_argument('-out', '--output', required=True, help='Transformed output feature file')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-r', '--reverse', action='store_true', help='Transform with reverse k-mers')
    options_group.add_argument('-c', '--complement', action='store_true', help='Transform with complementary k-mers')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug Console')

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

    logger.debug("Reading...")
    ids, counts = fileIO.read_feature_file(args.input, normalize=False)
    logger.debug("Transforming...")
    transformed_counts = transform_kmers(counts, reverse=args.reverse, complement=args.complement)
    logger.debug("Saving...")
    fileIO.save_counts(transformed_counts, ids, args.output, args=args, header="Transformed k-mer count file")
    logger.debug("Complete.")