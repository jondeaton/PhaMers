#!/usr/bin/env python
'''
kmers.py
This script is a python module for counting k-mers in biological sequence data.
It can also be used from the command line on fasta files:

python kmer.py my_sequences.fasta my_sequences_5mers.csv -k 5

'''

import numpy as np
import os
import gzip
import argparse
import time
import random
from Bio import SeqIO
import logging


__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DNA = 'ATGC'
RNA = 'AUGC'
protein = 'RHKDESTNQCUGPAVILMFYW'

def count_string(sequence, kmer_length, symbols=DNA, normalize=False):
    '''
    k-mer counting function
    :param sequence: The sequence to count k-mers in
    :param kmer_length: The length of the k-mer to count
    :param symbols: The symbols to count
    :param normalize: Normalize count vector by total number of k-mers
    :param verbose: Verbose output
    :return: A numpy array with the k-mer counts as integers or floats
    '''
    if len(symbols) < 10:
        # Integer replacement method (may only be used for kmer_length < 10)
        sequence = sequence_to_integers(sequence, symbols)
        num_symbols = len(symbols)
        kmer_count = np.zeros(pow(num_symbols, kmer_length), dtype=(int, float)[normalize])
        for i in xrange(len(sequence) - kmer_length + 1):
            integer_kmer = sequence[i:i + kmer_length]
            if '-' not in integer_kmer:
                kmer_count[int(integer_kmer, num_symbols)] += 1
    else:
        # Mathematical method (for any k)
        num_symbols = len(symbols)
        lookup = {char: symbols.index(char) for char in symbols}
        lookup.update({char.lower(): symbols.index(char) for char in symbols})
        kmer_count = np.zeros((1, pow(num_symbols, kmer_length)), dtype=(int, float)[normalize])

        reindex = True
        for i in xrange(len(sequence) - kmer_length + 1):
            if reindex:
                try:
                    kmer = sequence[i:i + kmer_length]
                    index = sum([lookup[kmer[n]] * pow(num_symbols, kmer_length - n - 1) for n in xrange(kmer_length)])
                    kmer_count[index] += 1
                    reindex = False
                except:
                    # Still encountering that bad character
                    pass
            else:
                try:
                    index = (index * num_symbols) % pow(num_symbols, kmer_length) + lookup[
                        sequence[i + kmer_length - 1]]
                    kmer_count[index] += 1
                except:
                    # Ran into a bad character
                    reindex = True
    if normalize and np.sum(kmer_count) > 0:
        kmer_count = normalize_counts(kmer_count)
    return kmer_count

def count(data, kmer_length, symbols=DNA, normalize=False):
    '''
    K-mer counting function
    :param data: Either a string sequence or a list of strings in which k-mer will be counted
    :param kmer_length: An integer specifying the length of k-mer to count (i.e. The 'k' in 'k-mer')
    :param symbols: The symbols counted as k-mers
    :param normalize: Boolean causing normalization of k-mer count vector by total number of k-mers present
    :return: A numpy array with shape = (num_sequences, num_symbols ^ kmer_length) with each element specifying the
    k-mer count or frequency of that k-mer. Mapping of k-mers to index is done lexicographically with hierarchy
    determined by the order of characters in the symbols argument. (i.e. 'AAAT' is at index 1 for DNA)
    '''
    if isinstance(data, list):
        # The data is a list of strings
        if len(data) == 1:
            # If the list only has one sequence in it
            return count(data[0], kmer_length, symbols=symbols, normalize=normalize)
        else:
            # If the list has multiple sequences in it
            kmer_count = np.zeros((len(data), pow(len(symbols), kmer_length)), dtype=(int, float)[normalize])
            i = 0
            for sequence in data:
                logger.info("Counting %d-mers in %d of %d... %s..." % (kmer_length, i+1, len(data), sequence[:kmer_length]))
                kmer_count[i, :] = count_string(sequence, kmer_length, symbols=symbols, normalize=normalize)
                i += 1
    elif isinstance(data, str):
        kmer_count = count_string(data, kmer_length, symbols=symbols, normalize=normalize)
    else:
        print "Data was not str or list: %s\n%s ..." % (type(data), data.__str__()[:25])
        kmer_count = None
    return kmer_count


def sequence_to_integers(sequence, symbols):
    '''
    Replace the characters of a sequence with their lexicographic enumeration id
    :param sequence: The sequence to convert
    :param symbols: The symbols to use for the lexicographic enumeration
    :return: A sequence with each symbol in the sequence replaced by it's index in symbols
    '''
    for no_read in set(sequence) - set(symbols):
        sequence = sequence.replace(no_read, '-')
    index = 0
    for symbol in symbols:
        sequence = sequence.replace(symbol, str(index))
        index += 1
    return sequence


def get_index(kmer, symbols):
    '''
    Debugging function used for getting lexicographic indicies of k-mers.
    :param kmer: The k-mer represented as String
    :param symbols: the symbols to use as a reference for lexicographic indexing
    :return: The lexicographic index of the k-mer
    '''
    return int(sequence_to_integers(kmer, symbols), len(kmer))


def normalize_counts(counts):
    '''
    This function normalizes a k-mer count array by the total number of k-mers
    :param counts: A numpy array with rows containing counts, can be 1D or 2D array
    :return: A numpy array with each row divided by the sum of each row
    '''
    counts = counts.astype(float)
    if len(counts.shape) == 1:
        counts = counts / np.sum(counts)
    else:
        for i in xrange(counts.shape[0]):
            counts[i, :] /= np.sum(counts[i, :])
    return counts


def count_file(input_file, kmer_length, symbols=DNA, normalize=False):
    '''
    This function counts k-mers from a fasta file
    :param input_file: The name or path to the fasta file to count k-mers in. Zipped files are ok!
    :param kmer_length: The length of k-mer to count as an integer
    :param symbols: A string containing the symbols to count
    :param normalize: Set this optional argument to True to normalize the k-mer count vectors to frequencies
    :param verbose: Set this optional argument to True for a verbose output
    :return: A tuple with the first element being a list of fasta headers, and the second element being a numpy array
        with rows corresponding to the k-mer counts of the sequences with the headers in the headers list
    '''
    try:
        ids = get_fasta_ids(input_file)
    except IOError:
        print "Bad file: %s" % os.path.basename(input_file)
        return None, None
    counts = np.zeros((len(ids), pow(len(symbols), kmer_length)), dtype=(int, float)[normalize])
    if input_file.endswith('.gz'):
        f = gzip.open(input_file, 'r')
    else:
        f = open(input_file, 'r')
    records = SeqIO.parse(f, 'fasta')
    i = 0
    for record in records:
        counts[i, :] = count(str(record.seq), kmer_length, symbols=symbols, normalize=normalize)
        i += 1
    return ids, counts


def count_directory(directory, kmer_length, identifier='fna',symbols=DNA, sum_file=True, sample=0):
    '''
    This function counts k-mers in all fasta files within a directory
    :param directory: A string specifying the directory containing the fasta sequences
    :param kmer_length: An integer specifying the k-mer length to count
    :param identifier: A string that is contained within the basename of all fasta files to be counted, default "fna"
    :param symbols: Optional argument specifying the symbols to be used in counting
    :param sum_file: An optional parameter that will result in summing of all k-mer counts in the files
    :param verbose: Verbose output
    :return: A tuple with the first element being a list of fasta headers, and the second element being a numpy array
        with rows corresponding to the k-mer counts of the sequences with the headers in the headers list
    '''
    selected_files = [os.path.join(directory, file) for file in os.listdir(directory) if identifier in os.path.basename(file)]
    if sample:
        random.shuffle(selected_files)
    ids = len(selected_files) * ['']
    counts = np.zeros((len(selected_files), pow(len(symbols), kmer_length)))
    i = 0
    done = False
    while not done:
        file = selected_files[i]
        file_ids, file_counts = count_file(file, kmer_length, symbols=symbols)
        if (file_ids == None and file_counts == None) or len(file_ids) == 0 or np.sum(file_counts) == 0:
            print "Bad file: %s" % file
            selected_files.remove(file)
            continue

        if sum_file and len(file_counts.shape) == 2:
            # Summing k-mer counts of all sequences within a file
            file_id = file_ids[0]
            file_counts = np.sum(file_counts, axis=0)

        ids[i] = file_id
        counts[i] = file_counts
        i += 1
        done = (sample and i == sample) or i == len(selected_files)

    return ids, counts


def kmers(k, symbols=DNA):
    '''
    This function returns all of the k-mers for a given k and set of symbols
    :param k: The length of the k-mer
    :param symbols: The symbols used for k-mers
    :return: A list containing strings representing all of the k-mers in lexicographic order
    '''
    singles = [symbol for symbol in symbols]
    return extend_mers(singles, k - 1, symbols)


def extend_mers(mers, k, symbols):
    '''
    I'm not really sure what this function does... but its required for the kmers function to work
    :param mers: A growing list of k-mers
    :param k: The length of the k-mer
    :param symbols: The symbols to use for kmers
    :return: Something closer to what you're trying to get at
    '''
    if k == 0:
        return mers
    extd_mers = []
    for base in symbols:
        extension = [''] * len(mers)
        for i in range(len(mers)):
            extension[i] = base + mers[i]
        extd_mers = extd_mers + extension
    return extend_mers(extd_mers, k - 1, symbols)


def read_fasta(fasta_file):
    '''
    Function for reading the contents of a fasta file
    :param fasta_file: the file name or path to file. Zipped files are okay.
    :return: A tuple containing a list of fasta id strings and a list of string sequences
    '''
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    ids, sequences = [], []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        ids.append(get_id(str(record.id)))
        sequences.append(str(record.seq))
    del records
    return np.array(ids), sequences


def get_fasta_ids(fasta_file):
    '''
    This file retrieves only the headers from a fasta file
    :param fasta_file: The fasta filename
    :return: A numpy array of the headers in that fasta file
    '''
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    ids = []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        ids.append(get_id(str(record.id)))
    del records
    return ids


def get_fasta_sequences(fasta_file):
    '''
    This file retrieves the sequences from a fasta file
    :param fasta_file:
    :return:
    '''
    if fasta_file.endswith('.gz'):
        f = gzip.open(fasta_file, 'r')
    else:
        f = open(fasta_file, 'r')
    sequences = []
    records = SeqIO.parse(f, 'fasta')
    for record in records:
        sequences.append(str(record.seq))
    return sequences


def read_kmer_file(kmer_file, normalize=False, id=None, old=False):
    '''
    A function for reading a k-mer file
    :param kmer_file: The file name of the k-mer file, the file should be in a csv format
    :param normalize: Set to true to normalize the k-mers read from file
    :param id: Get on ly the kmer count vector for a given ID
    :param old: For reading old-style kmer files
    :return: A numpy array containing the k-mer count data from that file
    '''
    if old:
        kmers = np.loadtxt(kmer_file, delimiter=',', dtype=(int, float)[normalize])
        ids = ['No_ID'] * kmers.shape[0]
    else:
        data = np.loadtxt(kmer_file, delimiter=',', dtype=str)
        ids = list(data[:, 0].transpose())
        kmers = data[:, 1:].astype(int)

    if normalize:
        kmers = normalize_counts(kmers)

    if id:
        return kmers[ids.index(id)]
    else:
        return ids, kmers


def read_headers(header_file):
    '''
    A function for reading a headers file
    :param header_file: The file name of the headers file
    :return: A list containing all of the headers in that file, in the same order
    '''
    return [line.strip() for line in open(header_file, 'r').readlines() if not line.startswith('#')]


def load_counts(kmer_length, location=None, counts_file=None, identifier='fna', normalize=False, symbols=DNA):
    '''
    This file is for loading hdeaders and k-mer counts from either a directory containing fasta files, a single fasta
    file, or a pre-counted headers and k-mer count file. This funciton will first check for pre-counted files and will
    then check to see if the "location" is a directory or a file to decide how to count it
    :param kmer_length: The k-mer length to count
    :param location: The directory containing fasta files OR fasta file
    :param headers: The filename of the headers file
    :param counts: The filename of the k-mer count file
    :param symbols: Symbols to use for counting
    :param verbose: Verbose output
    :return: A tuple containing a list of headers, and list of k-mer coutns as a numpy array, in that order
    '''
    if counts_file and os.path.isfile(counts_file):
        logger.info("Loading %d-mers from %s..." % (kmer_length, os.path.basename(counts_file)))
        tic = time.time()
        ids, counts = read_kmer_file(counts_file, normalize=normalize)
    elif location and os.path.isfile(location):
        logger.info("Counting %d-mers in %s..." % (kmer_length, os.path.basename(location)))
        tic = time.time()
        ids, counts = count_file(location, kmer_length, normalize=normalize,  symbols=symbols)
        if counts_file:
            save_counts(counts, ids, count_file)
    elif location and os.isdir(location):
        logger.info("Counting %d-mers in %s..." % (kmer_length, os.path.basename(location)))
        tic = time.time()
        ids, counts = count_directory(location, kmer_length, normalize=normalize, identifier=identifier, symbols=symbols)
        if counts_file:
            save_counts(counts, ids, count_file)
    run_time = time.time() - tic
    logger.info("done. %dhr %dmin %.1fsec" % (run_time // 3600, (run_time % 3600) // 60, run_time % 60))
    return ids, counts


def save_counts(counts, ids, file_name, args=None, header='k-mer count file'):
    '''
    A function for saving a k-mer count array
    :param counts: The numpy array containing the k-mer count data
    :param ids: The string ids for each sequence being represented in the kmer count array
    :param file_name: The name of the file to save the data to
    :param header: An optional parameter specifying the header of the file
    :return: None
    '''
    if args is not None:
        header += '\n%s\n%s' % (time.strftime("%Y-%m-%d %H:%M"), generate_summary(args).replace('\n#','\n'))
        logger.info("Saving counts as %s with file header:\n%s" % (os.path.basename(file_name), header))
    X = np.hstack((np.array([ids]).transpose(), counts.astype(int).astype(str)))
    np.savetxt(file_name, X, fmt='%s', delimiter=',', header=header)


def combine_kmer_header_files(kmer_file, header_file, new_file):
    '''
    This function is for combining old kmer-count formats into a single file
    :param kmer_file:
    :param header_file:
    :param new_file:
    :return: None
    '''
    headers = read_headers(header_file)
    ids = [get_id(header) for header in headers]
    kmers = np.loadtxt(kmer_file, delimiter=',', dtype=int)
    save_counts(kmers, ids, new_file)


def get_contig_id(header):
    '''
    Returns the ID number from a contig header
    :param header: The contig header
    :return: The contig ID number as an integer
    '''
    parts = header.split('_')
    return int(parts[1 + parts.index('ID')].replace('-circular', ''))


def get_bacteria_id(header):
    '''
    This function gets the ID from a bacteria fasta header
    :param header: The header string
    :return: The genbank id
    '''
    id = header.split(' ')[0]
    if is_genbank_id(id):
        return id
    id = header.split('\t')[1].replace('>',  '')
    if is_genbank_id(id):
        return id


def get_phage_id(header):
    '''
    This function gets the genbank id from the header of phage fasta
    :param header: The string fasta header
    :return: The string representation of the id
    '''
    return header.split('|')[3].replace('>', '')


def is_genbank_id(id):
    '''
    This function decides if a string is a genbank id
    :param id: The string to test
    :return: True if the string represents a genbank id format
    '''
    return not represents_float(id) and id[-2] == '.'


def represents_float(s):
    '''
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_id(header):
    '''
    Decides what kind of ID needs to be retrieved and decides how to correctly parse it
    :param header: the header to be parsed
    :return: The ID which has been retrieved from the header
    '''
    if '_ID_' in header:
        return get_contig_id(header)
    elif header.count('|') == 4:
        return get_phage_id(header)
    else:
        return get_bacteria_id(header)


def generate_summary(args):
    '''
    This makes a summary for the output file
    :param args: The parsed argument parser from function call
    :return: a beautiful summary
    '''
    return args.__str__().replace('Namespace(', '# ').replace(')', '').replace(', ', '\n# ').replace('=', ':\t')


if __name__ == '__main__':

    script_description = "This script counts k-mers in sequence data"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('input_file', type=str, help='Fasta compilation file to count k-mers in')
    input_group.add_argument('-id', '--file_identifier', type=str, default='.fna', help='File identifier if directory')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('output_file', type=str, help='Output CSV file of kmer counts')

    options_group = parser.add_argument_group('Options')
    parser.add_argument('-k', '--kmer_length', type=int, default=4, help='Length of kmer to count')
    options_group.add_argument('-s', '--sample', type=int, help='Number of sequences to sample and count')
    options_group.add_argument('-sym', '--symbols', type=str, default=DNA, help='Symbols to use in kmer counting')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug Console')
    args = parser.parse_args()

    input = args.input_file
    output = args.output_file
    kmer_length = args.kmer_length
    sample = args.sample
    file_identifier = args.file_identifier
    symbols = args.symbols

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')

    logger.info("Couting k-mers...")

    tic = time.time()
    if input and os.path.isfile(input):
        ids, kmers = count_file(input, kmer_length, symbols=symbols)
    elif input and os.path.isdir(input):
        ids, kmers = count_directory(input, kmer_length, symbols=symbols, identifier=file_identifier, sample=sample)
    else:
        logger.error("%d was not an acceptable file or directory" % input)
        exit()

    save_counts(kmers, ids, output, args=args)

    logger.info("done. %.1f seconds" % (time.time() - tic))
