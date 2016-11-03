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
    This function makes a list of indicies that
    :param k: The length of the k-mers (k)
    :param num_symbols: The number of symbols available
    :param reposition_indicies: A numpy array of indicies where the positions of each symbol should be mapped to
    when they are read in. For instance, to get the transformed indicies for couting k-mers in reverse, this parameter
    should be an array of integer indicies couting down in reverse (i.e. [3, 2, 1, 0] for k=4)
    :param index_substitution_map: This is a dictionary that maps the index of each symbol in the sylbols that were used
    to count the k-mers, to a differenc index. For example, to count k-mers in the complement of a DNA sequence
    where the symbol are 'ATGC', this parameter should be {0: 1, 1: 0, 2: 3, 3: 2}.
    :return: A list of indicies that corespond to where each k-mer index shoudl be re-mapped to
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
    This function will transform a set of DNA k-mer counts into a set of counts that would have been generated
     if reverse, complementary, or reverse complementary k-mers had been counted instead
    :param counts: A numpy array of k-mer count vectors
    :param reverse: Specify if you want the reverse counts instead
    :param complement: Specify if you want the complementary counts instead
    :param symbols: The symbols that were used to generate the k-mer count vectors
    :return: A numpy array of k-mer count vectors as though they were counted differencly
    """
    num_symbols = len(symbols)
    k = int(round(np.math.log(counts.shape[1], num_symbols)))
    counts = counts.transpose()
    if reverse and not complement:
        return counts[get_reverse_indicies(k)].transpose()
    elif complement and not reverse:
        return counts[get_DNA_complement_indicies(k)].transpose()
    elif reverse and complement:
        return counts[get_reverse_complement_indicies(k)].transpose()
    else:
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