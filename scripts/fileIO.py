#!/usr/bin/env python
"""
fileIO.py

This script gives save and load functionality for the various file types used in PhaMers
"""

import gzip
from Bio import SeqIO
import numpy as np
import basic
import id_parser
import kmer
import taxonomy
from analysis import phage
import logging
import transform_kmers

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_fasta(fasta_file):
    """
    Function for reading the contents of a fasta file
    :param fasta_file: the file name or path to file. Zipped files are okay.
    :return: A tuple containing a list of fasta id strings and a list of string sequences
    """
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    ids, sequences = [], []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        ids.append(id_parser.get_id(str(record.id)))
        sequences.append(str(record.seq))
    del records
    return np.array(ids), sequences


def get_fasta_ids(fasta_file):
    """
    This file retrieves only the headers from a fasta file
    :param fasta_file: The fasta filename
    :return: A numpy array of the headers in that fasta file
    """
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    ids = []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        ids.append(id_parser.get_id(str(record.id)))
    del records
    return np.array(ids)


def get_fasta_sequences(fasta_file):
    """
    This file retrieves the sequences from a fasta file
    :param fasta_file:
    :return:
    """
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    sequences = []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        sequences.append(str(record.seq))
    return sequences


# Lineage files
def read_lineage_file(lineage_file, extend=False):
    """
    Reads a lineage file that was created by retrieve_lineages, and returns data as a dictionary
    :param lineage_file: The filename of the lineage file
    :return: a dictionary mapping phage id to taxonomic lineage
    """
    dictionary = read_label_file(lineage_file)
    if extend:
        ids = dictionary.keys()
        lineages = taxonomy.extend_lineages(dictionary.values())
        dictionary = {ids[i]: lineages[i] for i in xrange(len(ids))}
    return dictionary


def read_label_file(label_file):
    """
    This function reads a file that encodes a mapping from id to label
    :param label_file: The file where the mapping information is stored
    :return: A dictonary that maps id to label
    """
    if not label_file:
        raise TypeError("No label file provided.")
    lines = open(label_file, 'r').readlines()[1:]
    dictionary = {line.split('\t')[0]: [kind.strip() for kind in line.split('\t')[1].strip().split(';')] for line in lines}
    return dictionary

# K-mer files and header files
def read_headers_file(header_file):
    """
    A function for reading a headers file
    :param header_file: The file name of the headers file
    :return: A list containing all of the headers in that file, in the same order
    """
    return [line.strip() for line in open(header_file, 'r').readlines() if not line.startswith('#')]


def read_feature_file(feature_file, normalize=False, id=None, old=False, transform=False):
    """
    A function for reading a k-mer file
    :param kmer_file: The file name of the k-mer file, the file should be in a csv format
    :param normalize: Set to true to normalize the features by row sum
    :param id: Get only the kmer count vector for a given ID
    :param old: For reading legacy formatted k-mer files without ids in the first columns
    :return: A numpy array containing the k-mer count data from that file
    """
    if old:
        features = np.loadtxt(feature_file, delimiter=',')
        ids = ['No_ID'] * features.shape[0]
    else:
        data = np.loadtxt(feature_file, delimiter=',', dtype=str)
        # Make into a 2D array if its a 1D array
        if len(data.shape) == 1:
            data = np.array([data])
        ids = list(data[:, 0].transpose())
        features = data[:, 1:].astype(int)

    if transform:
        features += transform_kmers.transform_kmers(features, reverse=True, complement=True)

    if normalize:
        features = kmer.normalize_counts(features)

    # Convert integer IDs from strings to integers, if possible
    ids = np.array(ids)

    if id:
        return features[ids == id]
    else:
        return ids, features


def save_counts(counts, ids, file_name, args=None, header='K-mer count file'):
    """
    A function for saving a k-mer count array
    :param counts: The numpy array containing the k-mer count data
    :param ids: The string ids for each sequence being represented in the kmer count array
    :param file_name: The name of the file to save the data to
    :param header: An optional parameter specifying the header of the file
    :return: None
    """
    if args is not None:
        header = basic.generate_summary(args, header=header)
    data = np.hstack((np.array([ids]).transpose(), counts.astype(int).astype(str)))
    np.savetxt(file_name, data, fmt='%s', delimiter=',', header=header)


def combine_kmer_header_files(kmer_file, header_file, new_file):
    """
    This function is for combining old kmer-count formats into a single file
    :param kmer_file:
    :param header_file:
    :param new_file:
    :return: None
    """
    headers = read_headers_file(header_file)
    ids = [id_parser.get_id(header) for header in headers]
    kmers = np.loadtxt(kmer_file, delimiter=',', dtype=int)
    save_counts(kmers, ids, new_file)


# t-SNE files
def save_tsne_data(filename, tsne_data, ids, args=None, chops=None):
    """
    This function saves t-SNE data to file with ids in the first column and x,y values in the second and third
    :param filename: The name of the file to save the data to
    :param tsne_data: x,y coordinates for each poitn to store
    :param ids: The ids corresponding to each data-point in the tsne_data
    :param header: The header of the file
    :return: None
    """
    header = "t-SNE coordinates file"
    if chops:
        header += "\nchops: %s" % chops.__str__().replace('(', '').replace(')', '').replace('[','').replace(']','').strip()
    if args:
        header = basic.generate_summary(args, header=header)
    data = np.hstack((np.array([ids]).transpose(), tsne_data.astype(str)))
    np.savetxt(filename, data, fmt='%s', delimiter=',', header=header)


def read_tsne_file(tsne_file):
    """
    This file returns the data from a t-SNE file, which can be split into the kinds that is
    :param tsne_file: The filename of the t-SNE csv file
    :param chop: Set to true to have the data split into a list based on the number of unknown, positive, and negative
    data-points. Number of each kind are specified on the second line of the file as follows: #unk,pos,neg=(n_unk, n_pos, n_neg)
    :return: Either the raw data in a numpy array, or if chopped, a list of numpy arrays in the order unknown, positive
    and then negative
    """
    if not tsne_file:
        raise ValueError("t-SNE file is None")
    data = np.loadtxt(tsne_file, dtype=str, delimiter=',')
    ids = list(data[:, 0].transpose())
    points = data[:, 1:].astype(float)
    chops = None
    with open(tsne_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#') and 'chops' in line:
                chops = line.split(":")[1].strip()
                chops = map(int, chops.split(','))
    return ids, points, chops


# Phamer file read/write
def save_phamer_scores(ids, scores, file_name, args=None):
    """
    This function saves phamer scores to a csv
    :param ids: List of ids
    :param scores: List of scores
    :param filename: Filename to save it to
    :return: None
    """
    header = "PhaMers score file"
    if args is not None:
        header = basic.generate_summary(args, header=header)
    arr = np.vstack((ids.astype(str), scores.astype(str))).transpose()
    np.savetxt(file_name, arr, delimiter=', ', header=header, comments="# ", fmt="%s")


def read_phamer_output(filename):
    """
    This file reads a phamer summary file and returns the scores for the Phamer run
    :param filename: The file name of the Phamer summary file
    :return: A distionary mapping contig ID to Phamer score
    """
    # todo: fix this... this is stupid
    score_dict = {}
    lines = open(filename, 'r').readlines()
    for line in lines:
        if '#' not in line:
            try:
                id = line.split(',')[0]
            except:
                id = id_parser.get_contig_id(line.split()[0])
            score_dict[id] = float(line.split()[1])
    return score_dict

# VirSorter
def read_virsorter_file(filename, dataset=''):
    """
    This function reads a VirSorter summary file and generates phage objects from the file
    :param filename: The name of the VirSorter summary file
    :param dataset: The name of the dataset that the VirSorter file was made from
    :return: A tuple containing phage objects that represent those predicted from the VirSorter run
    """
    if dataset == '':
        dataset = filename

    lines = open(filename, 'r').readlines()
    vs_phage = []
    for line in lines:
        if line.startswith('##'):
            if 'category' in line:
                kind = line.split('-')[1].strip()
                category = int(line.split('-')[2].strip().split()[1])
        else:
            vs_phage.append(phage(kind, category, dataset, line))
    return vs_phage


if __name__ == '__main__':
    print "This is a python module, not meant to be run from the command line."



