#!/usr/bin/env python

import os
import sys
import time
from Bio import Entrez
from Bio import SeqIO
import datetime
import argparse
import numpy as np
from scipy import stats
import logging

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Just for reference
hierarchy = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
viral_hierarchy = ['Baltimore', 'Order', 'Family', 'Sub-family', 'Genus']

# If you use this script, enter your email here
__email__ = "jdeaton@stanford.edu"

Entrez.email = __email__
wait_time = 1 / 3.0

def change_email(email):
    """
    This function changes the email used to access the NCBI database
    :param email: The string email to use
    :return: None
    """
    __email__ = email
    Entrez.email = __email__


# Fucntions that request information from the NCBI servers
def get_tax_id(gb_id):
    """
    This function finds the Entrez taxonomy ID for a "gb" ID by a taxonomy database request
    :param gb_id: The gb ID of the sequence to search for
    :return: The taxonomy database ID "taxid"
    """
    handle = Entrez.efetch(db='nucleotide', id=gb_id, rettype='gb', retmode='text')
    field = 'db_xref="taxon:'
    end = '//'
    line = ''
    while field not in line and line != end:
        line = handle.readline()

    if line == end:
        logger.warning('Taxonomy id not found for Entrez search: %s' % gb_id)
        return ''
    else:
        return line[line.index(":")+1:-2]


def get_taxonomy(tax_id='', gb_id='', wait=True):
    """
    This function gets the content of a NCBI taxonomy database in xml format
    :param tax_id: NCBI taxid
    :param gb_id: NCBI gb ID
    :param wait: Set to False if you want to ignore the "3 request per second" Entrez rule
    :return: The content of a NCBI taxonomy database page in xml format
    """
    if not tax_id == '':
        search = Entrez.efetch(id=tax_id, db='taxonomy', retmode='xml')
        return Entrez.read(search)
    elif not gb_id == '':
        tax_id = get_tax_id(gb_id)
        time.sleep(wait * wait_time)
        return get_taxonomy(tax_id=tax_id)
    else:
        return ''


def get_lineage(taxonomy='', tax_id='', gb_id='', wait=True):
    """
    This function returns the taxonomic lineage of a species from a taxonomy report or taxonomy/gb id
    :param taxonomy: The string representation of the taxonomy page
    :param tax_id: The taxonomy id
    :param gb_id: The GenBank ID
    :param wait: Set to False if you want to ignore the "3 request per second" Entrez rule
    :return: The lineage of a species as a list
    """
    if not taxonomy == '':
        return taxonomy[0][u'Lineage'].strip().split(';')
    elif not tax_id == '':
        return get_lineage(taxonomy=get_taxonomy(tax_id = tax_id, wait=wait), wait=wait)
    elif not gb_id == '':
        return get_lineage(taxonomy=get_taxonomy(gb_id = gb_id, wait=wait), wait=wait)
    else:
        return 'lineage not found'


# Other functions
def string_lineage(lineage):
    """
    Returns the string representaton of a lineage list
    :param lineage: A list of taxonomic classifications
    :return: A single string made by joining all of the elements of the list by a semicolon
    """
    return '; '.join(lineage)


def extend_lineages(lineages):
    """
    This function takes a set of lineages and makes all lineages the length of the longest lineage in the list
    :param lineages: A list of lineage lists
    :return: A list with each lineage extended to the be the length of the longest lineage in the list
    """
    max_classification = deepest_classification(lineages)
    for i in xrange(len(lineages)):
        lineage = lineages[i]
        lineages[i] = np.concatenate((lineage, list(np.repeat(lineage[-1], max_classification - len(lineage)))))
    return lineages


def retrieve_lineages(ids, lineage_file, wait=True):
    """
    Retrieves all lineages for a list of gb ids and write them to file
    :param ids: The list of ids to retrieve
    :param lineage_file: The file to write the lineages to
    :param wait: Set to False if you want to ignore the "3 request per second" Entrez rule
    :return: None
    """
    if wait:
        logger.warning("Wait time: %.3f sec. ~run time: %d m %d s" % (2 * wait_time, (len(ids)*2* wait_time) // 60, (len(ids) * 2 * wait_time) % 60))
    else:
        logger.warning('Executed disregarding Entrez \"3 requests per second limit\" rule.')

    f = open(lineage_file, 'w')
    f.write("#Lineage file. %s\n" % datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    i = 1
    for id in ids:
        time.sleep(wait * wait_time)
        lineage = string_lineage(get_lineage(gb_id=id, wait=wait))
        f.write("%s\t%s\n" % (id, lineage))
        logger.info("Retrieved %s (%d of %d) " % (id, i, len(ids)))
        i += 1
    logger.info("Completed.")


def lineage_proportions(lineages, normalize=True):
    """
    This function finds the proportions of each lineage classificasion from a list of lineages
    :param lineages: A list of lists that represents a set of lineages
    :return: A list of dictionaries that map taxonomic classification to proportion within of set
    """
    num_lineages = len(lineages)
    proportions = []
    lineages = extend_lineages(lineages)
    for lineage_depth in xrange(deepest_classification(lineages)):
        mapping = {}
        kinds = [lineage[lineage_depth] for lineage in lineages]
        diversity = set(kinds)
        for kind in diversity:
            mapping[kind] = kinds.count(kind)
        if normalize:
            mapping[kind] /= float(num_lineages)
        proportions.append(mapping)
    return proportions


def test_population_lineage_differenses(test_lineages, base_lineages):
    """

    :param test_lineages:
    :param base_lineages:
    :return: A list of dictionaries that map classification to tuples of p_value and proportion
    """
    test_proportions = lineage_proportions(test_lineages, normalize=False)
    base_proportions = lineage_proportions(base_lineages, normalize=False)

    all_dictionaries = []

    for depth in xrange(len(test_proportions)):
        test_mapping = test_proportions[depth]
        base_mapping = base_proportions[depth]
        test_neg = np.sum(test_mapping.values())
        base_neg = np.sum(base_mapping.values())
        dictionary = {}
        for kind in test_mapping.keys():
            test_pos = test_mapping[kind]
            base_pos = base_mapping[kind]
            x = np.array([[base_neg - base_pos, base_pos], [test_neg - test_pos, test_pos]])

            if x[0, 0] == 0 or x[0, 1] == 0:
                results = (1, 1, 0, np.zeros((2,2)))
            else:
                results = stats.chi2_contingency(x)

            logger.info("test: %s" % x.__str__())
            logger.info("Results: %s" % results.__str__())

            dictionary[kind] = results
        all_dictionaries.append(dictionary)

    return all_dictionaries


def find_enriched_classification(test_lineages, base_lineages, depth):
    """
    This function takes a set of lineages and compares them to another set of of lineages at a given depths and
    determines if any classification at that depth is significantly enriched in the test lineage. It will be counted
    as enriched if a classification in test_lineages at the depth are more the majority, if it is larger than the
    proportions in base differences by a statistically significant amount. This is done with a 2 proportion test.
    :param test_lineages: A list of lineages to test for being enriched
    :param base_lineages: A list of lineages to compare to
    :param depth: The depth of classification to compare to
    :return: A tuple of classification, chi2_contingency test results, and the percentage, or None, None,
    None if there is not enrichment for any taxa
    """
    test_kinds = [test_lineages[i][depth] for i in xrange(len(test_lineages))]
    if len(test_lineages) == 0:
        return None, None, None
    all_kinds = [base_lineages[i][depth] for i in xrange(len(base_lineages))]
    diversity = set(all_kinds)
    for kind in diversity:
        base_count = all_kinds.count(kind)
        test_count = test_kinds.count(kind)
        base_total = len(all_kinds)
        test_total = len(test_kinds)
        test_ratio = float(test_count) / test_total
        base_ratio = float(base_count) / base_total

        x = np.array([[test_total - test_count, test_count], [base_total - base_count, base_count]])
        if x[1, 0] == 0:
            result = (1, 1, 0, np.zeros((2, 2)))
        else:
            result = stats.chi2_contingency(x)

        if result[1] <= 0.05 and test_ratio >= 0.5 and test_ratio > base_ratio:
            #logger.debug("%.1f%% %s p = %.2g" % (100*test_ratio, kind, result[1]))
            return kind, result, test_ratio

    return None, None, None


def deepest_classification(lineages):
    """
    This function finds the number of the deepest classification for a set of lineages
    :param lineages: A list of lists that represents a set of lineages
    :return: An integer representing the number of elements in the longest lineage in the list
    """
    return max([len(lineage) for lineage in lineages])


def make_lineage_file(fasta_file, lineage_file, wait=True):
    """
    This function makes a file that stores the mapping of GenBank IDs to taxonomic lineages
    :param fasta_file: The fasta file that contains all of the sequences to look up lineages for and also has the
    GenBank ID number as the fourth element of the fasta headers when the header is split by the pipe symbol
    :param lineage_file: The name of the file to save the lineages to
    :param wait: Set to False if you want to ignore the "3 request per second" Entrez rule
    :return: None
    """
    ids = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        id = seq_record.id.split('|')[3]
        ids.append(id)
        logger.info("Found gb id: %s" % id)

    logger.info("Read %d gb_ids from %s" % (len(ids),  os.path.basename(fasta_file)))
    logger.info("Writing lineages to %s" % os.path.basename(lineage_file))

    retrieve_lineages(ids, lineage_file, wait=wait)


if __name__ == '__main__':

    script_description = "This script creates a taxonomic lineage file that maps a fasta file with GB IDs to taxonomic lineages"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    inputs_group = parser.add_argument_group("Inputs")
    inputs_group.add_argument('fasta_file', type=str, help='Fasta file to find lineages from')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('lineage_file', type=str, help='Output lineage file')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-e', '--email', type=str, default=__email__, help='Entrez email')
    options_group.add_argument('-w', '--no_wait', action='store_true')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')

    args = parser.parse_args()

    change_email(args.email)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')

    logger.info("Making lineage file: %s" % os.path.basename(args.lineage_file))
    make_lineage_file(args.fasta_file, args.lineage_file, wait=not args.no_wait)
    logger.info("Completed making lineage file")