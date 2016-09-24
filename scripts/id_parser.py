#!/usr/bin/env python
'''
This script is meant to be used to parse ids from files
'''

import basic
import logging

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_contig_id(header):
    '''
    Returns the ID number from a contig header
    :param header: The contig header
    :return: The contig ID number
    '''
    header = header.strip().replace('>', '')
    try:
        parts = header.split('_')
    except:
        logger.warning("Invalid contig header: %s" % header)
        return header
    return int(parts[1 + parts.index('ID')].replace('-circular', ''))


def get_contig_name(header):
    '''
    This function gets the name of the contig, which is the number after "SuperContig_" in the contig header
    :param header: The contig header
    :return: The integer name of the contig
    '''
    header = header.strip().replace('>', '')
    parts = header.split('_')
    try:
        return int(parts[1 + parts.index('SuperContig')])
    except:
        logger.warning("Couldn't find name from: %s" % header)


def get_contig_length(header):
    '''
    This function finds the length of a contig given it's header
    :param header: The fasta header of the contig
    :return: The integer length of contig in base pairs
    '''
    header = header.strip().replace('>', '')
    parts = header.split('_')
    return int(parts[1 + parts.index('length')])


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
    return not basic.represents_float(id) and id[-2] == '.'


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


if __name__ == '__main__':
    print "This is a module, not meant to be run from the command line."
