#!/usr/bin/env python
"""
This script is used to parse files from JGI's Integrated Microbial Genomes (IMG) gene annotation pipeline
"""

import os
import time
import argparse
import logging
import basic

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class contig():
    """
    This class is for contigs
    """
    def __init__(self, id, genes=None):
        self.id = id
        if genes:
            self.genes = genes
        else:
            self.genes = {}

    def __str__(self):
        return '\n'.join([str(gene) for gene in self.genes.values()])


class gene():
    """
    This class is for genes that have been predicted by Integrated Microbial Genomes (IMG) gene annotation pipeline
    """
    def __init__(self, ga_id='Missing', contig_id=-1, id=-1, product='Missing_Product', phylogeny='Missing_Phylogeny'):
        self.ga_id = ga_id
        self.contig_id = contig_id
        self.id = id
        self.product = product
        self.phylogeny = phylogeny

    def __str__(self):
        return ', '.join(map(str, (self.contig_id, self.id, self.product, self.phylogeny)))

# Parsing all files
def get_contig_map(IMG_directory, contig_name_id_map, contig_ids=None, keyword=None):
    """
    This function parses files from a directory that contains a IMG COG, phylogeny, and product files and writes the
    compiled data of contig id, phylogenies, and produces a mapping of id to contig object.
    :param IMG_directory: The directory that contains IMG files
    :param contig_ids: A list of contig IDs to search for
    :param keyword: Pass a string to this function so that only phylogenies with this keyword are used
    :return: A map of contig ID to contig object
    """
    cog_file = basic.search_for_file(IMG_directory, end='.COG', first=True)
    phylo_file = basic.search_for_file(IMG_directory, end='.phylodist', first=True)
    product_file = basic.search_for_file(IMG_directory, end='.product.names', first=True)

    logger.info("Parsing IMG file: %s ..." % os.path.basename(cog_file))
    contig_map = parse_cog_file(cog_file, contig_name_id_map, contig_ids=contig_ids)
    logger.info("Parsing IMG file: %s ..." % os.path.basename(phylo_file))
    parse_phylodist(phylo_file, contig_map, contig_name_id_map, contig_ids=contig_ids, keyword=keyword)
    logger.info("Parsing IMG file: %s ..." % os.path.basename(product_file))
    parse_products(product_file, contig_map, contig_name_id_map, contig_ids=contig_ids)
    return contig_map


def parse_cog_file(cog_file, contig_name_map, contig_ids=None):
    """
    This function parses a cog file and returns a dictionary that maps contig id to congig objects
    :param cog_file: The file to parse
    :param contig_ids: A set of contig ids to search for, in case you didn't want to parse all of them
    :return: A list of gene objects that contain Ga ids, contig ids and gene numbers
    """
    with open(cog_file, 'r') as f:
        lines = f.readlines()
        f.close()
    contig_map = {}
    for line in lines:
        ga_string = line.split()[0]
        ga_id, contig_id, id = parse_Ga_string(ga_string, contig_name_map)
        if contig_ids is None or contig_id in contig_ids:
            try:
                the_contig = contig_map[contig_id]
                # If not error, then this contig already is in the dictionary
            except KeyError:
                # This is a new contig, so make a new object for it
                the_contig = contig(contig_id)
                contig_map[contig_id] = the_contig
            the_contig.genes[id] = gene(ga_id=ga_id, contig_id=contig_id, id=id)
    return contig_map


def parse_phylodist(phylodist_file, contig_map, contig_name_id_map, contig_ids=None, keyword=None):
    """
    This function reads a phylogeny prediction file and puts phylogenies into the appropriate fields of the contig objects
    that are passed in withint the contig_map parameter
    :param phylodist_file: An IMG phylogeny prediction file
    :param contig_map: a dictionary that maps contig ID to contig objects, each of which has a gene dictionary
    :param contig_name_id_map: A dictionary mapping contig names to congig ids
    :param contig_ids: A list of contig IDs to look for in the data
    :param keyword: Pass a string to this keyworded argument to only add phylogenies that contain that keyword
    :return: A dictionary that maps contig ID to phylogenies as a string format
    """
    lines = open(phylodist_file, 'r').readlines()
    for line in lines:
        ga_string = line.split()[0]
        ga_id, contig_id, id = parse_Ga_string(ga_string, contig_name_id_map)
        phylogeny = line.split()[4]
        if contig_ids is None or contig_id in contig_ids:
            the_contig = None
            try:
                the_contig = contig_map[contig_id]
            except KeyError:
                # This is a new contig, so make a new contig object
                the_contig = contig(contig_id)
                contig_map[contig_id] = the_contig
                the_contig.genes[id] = gene(ga_id=ga_id, contig_id=contig_id, id=id)

            try:
                if the_contig is not None:
                    # found a contig... add a phylogeny to the gene at the specified gene id
                    the_contig.genes[id].phylogeny = phylogeny
            except KeyError:
                # There was not a gene for that id already, so add the gene and phylogeny
                new_gene = gene(contig_id=contig_id, id=id, phylogeny=phylogeny)
                the_contig.genes[id] = new_gene


def parse_products(products_file, contig_map, contig_name_id_map, contig_ids=None):
    """
    This function parses an IMG product prediction file and places the products into the appropriate fields within the
    contig objects that are passed through the contig map
    :param products_file: An IMG  product prediction file
    :param contig_map: a dictionary that maps contig ID to contig objects, each of which has a gene dictionary
    :param contig_name_id_map: A dictionary mapping contig names to congig ids
    :param contig_ids: A list of contig ids to search for within the file
    :return: A dictionary
    """
    lines = open(products_file, 'r').readlines()
    for line in lines:
        ga_string = line.split()[0]
        ga_id, contig_id, id = parse_Ga_string(ga_string, contig_name_id_map)
        if contig_ids is None or contig_id in contig_ids:
            product = line.split('\t')[1]
            the_contig = None
            try:
                the_contig = contig_map[contig_id]
            except KeyError:
                the_contig = contig(contig_id)
                contig_map[contig_id] = the_contig
                the_contig.genes[id] = gene(ga_id=ga_id, contig_id=contig_id, id=id)

            try:
                if the_contig is not None:
                    # found this contig... add product to the specified id
                    the_contig.genes[id].product = product
            except KeyError:
                #There was not a gene for that id already, so add the gene and it's product
                new_gene = gene(contig_id=contig_id, id=id, product=product)
                the_contig.genes[id] = new_gene


def make_gene_csv(IMG_directory, output_filename, contig_name_id_map, contig_ids=None, keyword=None):
    """
    This function parses files from a directory that contains a IMG COG, phylogeny, and product files and writes the
    compiled data of contig id, phylogenies, and products into a tab-separated summary file
    :param IMG_directory: The directory that contains IMG files
    :param output_filename: The filename of the output summary file
    :param contig_ids: A list of contig IDs to search for
    :param keyword: Pass a string to this function so that only phylogenies with this keyword are used
    :return: None
    """
    contig_map = get_contig_map(IMG_directory, contig_name_id_map, contig_ids=contig_ids, keyword=keyword)
    f = open(output_filename, 'w')
    header = gene_csv_header(IMG_directory)
    f.write(header)
    for contig_id in contig_map:
        next_contig = contig_map[contig_id]
        f.write(str(next_contig) + '\n')


def parse_Ga_string(Ga_string, contig_name_id_map):
    """
    This function parses the string in the first column of Integrated Microbial Genomes (IMG) gene prediction files.
    :param Ga_string: The string in the first column of IMG gene prediction files
    :param contig_name_id_map: A map from contig name to contig id
    :return: A tuple containing the Ga id in the fist element, and the contig id as the second element,
    and an integer representing the id number of the id within the contig
    """
    splitted = Ga_string.split('_')
    Ga_id = splitted[0]
    starts = [1, 2, 0]
    stops = [-1, -2, 0, -3]
    for start in starts:
        for stop in stops:
            if stop == 0:
                contig_name = splitted[1][start:]
            else:
                contig_name = splitted[1][start:stop]
            try:
                contig_id = contig_name_id_map[contig_name]
                id = splitted[1][stop:]
                return Ga_id, contig_id, id
            except KeyError:
                pass
    exit("Error parsing: %s" % Ga_string)


def gene_csv_header(IMG_directory):
    """
    This function returns the header for a IMG compilation file
    :param IMG_directory: The directory used to generate the summary
    :return: A header that can be written at the top of an IMG summary file
    """
    cog_file = basic.search_for_file(IMG_directory, end='.COG', first=True)
    phylo_file = basic.search_for_file(IMG_directory, end='.phylodist', first=True)
    product_file = basic.search_for_file(IMG_directory, end='.product.names', first=True)
    now = time.strftime("%Y-%m-%d %H:%M")

    header = ''
    header += '# Integrated Microbial Genomes (IMG) gene prediction data\n'
    header += '# Directory: %s\n' % IMG_directory
    header += '# COG: %s\n' % os.path.basename(cog_file)
    header += '# Products: %s\n' % os.path.basename(product_file)
    header += '# Phylogeny: %s\n' % os.path.basename(phylo_file)
    header += '# Created: %s\n' % now
    header += '# contig ID, gene number, product name, phylogeny\n'
    return header


def string_phylogeny_content(phylogenies):
    """
    This function generates a string representation showing the percentile of the phylogenies provided that
    have the most common classification, for each phylogenetic depth
    :param phylogenies: A list of phylogenies, the elements of which are list of phylogenetic classifications (strings)
    :return: A string that shows the percentage of all phylogenies that are classified with the most common
    phylogeny for that classification deth.
    """
    phylogeny_percentages = phlogeny_percents(phylogenies)
    str_out = ''
    for tup in phylogeny_percentages:
        ratio, mode = tup
        str_out += "%.1f%% %s " % (100*ratio, mode)
    return str_out


def phlogeny_percents(phylogenies):
    """
    This function makes a list of tuples that contain the proportion of phylogenies that are the mode of
    that classification depth
    :param phylogenies: A list of phylogenies, each of which are a list of classifications
    :return: A list of tuples each of which contain the proportion of phylogenic classifications in phylogenies at
    the depth that are the mode of the phylogenies at that depth, in the first element, and the mode of the
    phylogenies at the depth in the second element of the tuple. The depth is given by the
    index of the tuple in the list.
    """
    num_genes = len(phylogenies)
    phylogeny_percentages = []
    max_depth = max([len(x) for x in phylogenies])
    for depth in xrange(max_depth):
        kinds = []
        for i in xrange(num_genes):
            try:
                kinds.append(phylogenies[i][depth])
            except:
                pass
        mode = basic.list_mode(kinds)
        ratio = kinds.count(mode) / float(num_genes)
        phylogeny_percentages.append((ratio, mode))
    return phylogeny_percentages

if __name__ == '__main__':

    script_description = 'This script parses IMG files'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('summary_file', type=str, help='IMG gene prediction summary file')
    input_group.add_argument('-id', '--contig_id', default=0, type=int, help='Contig ID')
    args = parser.parse_args()

    summary_file = args.summary_file
    contig_id = args.contig_id
    lines = [line for line in open(summary_file, 'r').readlines() if line.startswith(str(contig_id) + ',')]


    # todo: this is weird and should be fixed... maybe pipe it into a file
    print "Gene Predictions: "
    i = 1
    phylogenies = []
    for line in lines:
        product = line.split(', ')[2]
        phylogeny = line.split(', ')[3]
        phylogenies.append(phylogeny.split(';'))
        print "%d. %s" % (i, product),
        if not 'Missing' in phylogeny and ('irus' in phylogeny or 'hage' in phylogeny or 'iral' in phylogeny):
            print " (Viral)"
        else:
            print ''
        i += 1

    print string_phylogeny_content(phylogenies)



