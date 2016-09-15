#!/usr/bin/env python
'''

analysis.py

This script is for doing analysis of Phamer results and integrating results with VirSroter and IMG outputs
'''

import warnings
import os
import sys
import argparse
from Bio import Entrez
from Bio import SeqIO
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import numpy as np
import logging
import taxonomy as tax
import img_parser as img
import phamer
import kmer
import distinguishable_colors as dc
import matplotlib
matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
import pandas as pd
import cross_validate as cv

pd.options.mode.chained_assignment = None

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# If you use this script enter your email here
__email__ = "jdeaton@stanford.edu"
Entrez.email = __email__

class phage():
    '''
    This class is used to store data about putative phage predicted by VirSorter
    '''
    def __init__(self, kind, category, dataset, line):
        self.kind = kind
        self.category = category
        self.dataset = dataset

        self.header = line.split(',')[0]
        self.id = get_contig_id(self.header)
        self.contig_name = get_contig_name(self.header)
        self.length = get_contig_length(self.header)
        self.num_genes = int(line.split(',')[1])
        self.phamer_score = 0
        self.genes = []

    def __str__(self):
        top = "Phage Contig Name/ID:\t%d/%d\nLength:\t%.1f kbp\nVirSorter Classification:\t%s - " \
              "%s\nPhamer Score:\t%.3f\nGenes:\t%d\n" % (
            self.contig_name, self.id, self.length / 1000.0, self.kind, self.category, self.phamer_score,
            len(self.genes))

        for gene in self.genes:
            top += "%s\n" % gene.__str__()
        return top


def lineage_colors(lineages):
    '''
    This function makes a mapping of taxonomic classification to color so that multiple plots can be colored
    consistently
    :param lineages: A list of taxonomic lineages to make a color mapping for
    :return: A map of taxonomic classification to color. The color is a 1x3 numpy array representing a RGB color
    '''
    diversity = set()
    for depth in xrange(tax.deepest_classification(lineages)):
        diversity |= set([lineage[depth] for lineage in lineages])

    diversity = list(diversity)
    colors = dc.get_colors(len(diversity))
    color_map = {diversity[i]:colors[i] for i in xrange(len(diversity))}
    return color_map


def read_virsorter_file(filename, dataset=''):
    '''
    This function reads a VirSorter summary file and generates phage objects from the file
    :param filename: The name of the VirSorter summary file
    :param dataset: The name of the dataset that the VirSorter file was made from
    :return: A tuple containing phage objects that represent those predicted from the VirSorter run
    '''
    if dataset == '':
        dataset = filename

    lines = open(filename, 'r').readlines()
    vs_phage = []
    for line in lines:
        if line.startswith('##'):
            if 'category' in line:
                kind = line.split('-')[1].strip()
                category = line.split('-')[2].strip()
        else:
            vs_phage.append(phage(kind, category, dataset, line))
    return vs_phage


def get_category_dictionary(virsorter_summary):
    '''
    This function takes a VirSroter summary file and creates a dictionary that maps contig ID to VirSorter category
    :param virsorter_summary: The VirSorter summary file name as a string
    :return: A dictionary that mapping contig ID to VirSorter category
    '''
    lines = open(virsorter_summary, 'r').readlines()
    category_map = {}
    category = 0
    for line in lines:
        if line.startswith('##') and 'category' in line:
            category = int(line[3])
        elif not line.startswith('##'):
            id = get_contig_id(line.split(',')[0])
            category_map[id] = category
    return category_map


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


def get_contig_length(header):
    '''
    This function finds the length of a contig given it's header
    :param header: The fasta header of the contig
    :return: The integer length of contig in base pairs
    '''
    header = header.strip().replace('>', '')
    parts = header.split('_')
    return int(parts[1 + parts.index('length')])


def get_contig_name_map(contigs_file=None, headers=None):
    '''
    This file creates a dictionary that maps contig name to contig id
    :param headers: A list of contig headers
    :return: A map from contig name to id
    '''
    if contigs_file:
        headers = [line for line in open(contigs_file, 'r').readlines() if line.startswith('>')]

    map = {}
    for header in headers:
        name = get_contig_name(header)
        id = get_contig_id(header)
        map[name] = id
    return map


def map_id_to_header(contigs_file):
    '''
    This function makes a mapping from Contig ID to contig header
    :param contigs_file: The contigs file containing all contigs with headers starting with '>'
    :return: A dictionary that maps contig id to contig header
    '''
    headers = [line[1:] for line in open(contigs_file, 'r').readlines() if line.startswith('>')]
    map = {get_contig_id(header): header for header in headers}
    return map


def category_name(category):
    '''
    This function returns the qualitative confidence that a phage of a certain category is a phage
    :param category: The integer from 1 to 3 that represents the category of VirSorter prediction
    :return: The qualitative confidence that a phage of a certain category is a phage
    '''
    category_names = ['unknown', 'Complete - "pretty sure"', 'Complete - "quite sure"', 'Complete - "not so sure"']
    category_names += ['Prophages - "pretty sure"', 'Prophages - "quite sure"', 'Prophages -"not so sure"']
    return category_names[category]


def get_truth_table(phamer_summary, virsorter_summary, threshold=0):
    '''
    This function finds the true positives, false positives, false negatives, and true positives for phage
    predictions from phamer and from VirSorter
    :param phamer_summary: The phamer summary file to compare
    :param virsorter_summary: The VirSorter summary file to compare
    :param threshold: The threshold Phamer score to count as a prediction
    :return: A tuple of lists of IDs for true positives, false positives, false negatives, and true positives in that order
    '''
    phamer_dict = phamer.read_phamer_output(phamer_summary)
    phamer_ids = [id for id in phamer_dict.keys() if phamer_dict[id] >= threshold]

    vs_phage = read_virsorter_file(virsorter_summary)
    for phage in vs_phage:
        if phage.id in phamer_dict.keys():
            phage.phamer_score = phamer_dict[phage.id]
    vs_ids = [phage.id for phage in vs_phage]

    true_positives = [id for id in phamer_dict.keys() if id in vs_ids and id in phamer_ids]
    false_positives = [id for id in phamer_dict.keys() if id in phamer_ids and id not in vs_ids]
    false_negatives = [id for id in phamer_dict.keys() if id in vs_ids and id not in phamer_ids]
    true_negatives = [id for id in phamer_dict.keys() if id not in phamer_ids and id not in vs_ids]

    logger.info("%d true positives" % len(true_positives))
    logger.info("%d false positives" % len(false_positives))
    logger.info("%d false negatives" % len(false_negatives))
    logger.info("%d true negatives" % len(true_negatives))

    return true_positives, false_positives, false_negatives, true_negatives


def get_gene_product_dict(genbank_file):
    '''
    This function finds and returns a mapping of gene name to gene product name from a GenBank file containing only a
    single gene product
    :param genbank_file: The GenBank file to analyze
    :return: A dictionary that maps gene name to gene product name for all genes in the file
    '''
    gene_product_dict = {}
    gen = SeqIO.parse(genbank_file, 'genbank')
    for record in gen:
        for feature in record.features:
            if feature.type == 'CDS':
                name = feature.qualifiers['gene'][0]
                product = feature.qualifiers['product'][0]
                logger.info("%s: %s" % (name, product))
                gene_product_dict[name] = product
    return gene_product_dict


def prepare_for_SnapGene(genbank_file, destination):
    '''
    This function prepares a GenBank file from VirSorter to be viewed by SnapGene so that the product names are
    displayed instead of the gene names. (i.e. 'hypothetical protein' instead of 'gene_1')
    :param genbank_file: The output file from VirSorter that shows predicted phage in the GenBank format
    :param destination: The filename of the new changed file
    :return: None
    '''
    contents = open(genbank_file, 'r').read()
    if not contents == '':
        putative_phage_gb = contents.split('//')
        putative_phage_gb = [ppgb for ppgb in putative_phage_gb if not ppgb.strip() == '']
    else:
        putative_phage_gb = []

    for phage_gb in putative_phage_gb:
        contig_header = phage_gb.strip().split('\n')[0].split()[1]
        logger.info('Contig: %s' % contig_header)
        filename = 'VS_SuperContig_%d_ID_%d.gb' % (get_contig_name(contig_header), get_contig_id(contig_header))
        new_file = os.path.join(destination, filename)
        f = open(new_file, 'w')
        f.write(phage_gb)
        f.close()
        gene_prod_dict = get_gene_product_dict(new_file)

        f = open(new_file, 'r')
        contents = f.read()
        f.close()

        for gene in gene_prod_dict.keys():
            contents = contents.replace('/gene="%s"' % gene, '/gene="%s"' % gene_prod_dict[gene])
        contents = contents.strip() + '\n//'

        f = open(new_file, 'w')
        f.write(contents)
        f.close()


def make_lineage_plot(point, score, phage_kmers, phage_lineages, out_file='lineage_prediction.svg', color_map=None,
                      category=0, id=0):
    '''
    This function makes several lineage pie charts for a k-mer count datapoint based on nearby phage points
    :param point: The k-mer count datapoint as a numpy array
    :param phage_kmers: A numpy array containing the k-mer count
    :param phage_lineages: List of phage lineages that is in the SAME ORDER AS THE POINT IN phage_kmers
    :param filename: The name of the file where this breathtaking plot will be saved
    :return: None... just makes a savage plot
    '''
    num_phage = phage_kmers.shape[0]
    appended_data = np.append(phage_kmers, np.array([point]), axis=0)
    asmt = phamer.kmeans(appended_data, 86)
    #asmt = phamer.dbscan(appended_data, eps=0.012, min_samples=5)

    phage_lineages = tax.extend_lineages(phage_lineages)

    cluster = asmt[-1]
    cluster_phage = np.arange(num_phage)[asmt[:-1] == cluster]
    cluster_size = len(cluster_phage)
    cluster_lineages = [phage_lineages[i] for i in cluster_phage]
    titles = ['Baltimore Classification', 'Order', 'Family', 'Sub-Family', 'Genus']

    plt.figure(figsize=(18, 8))
    for lineage_depth in xrange(1, 6):
        ax = plt.subplot(2, 3, lineage_depth)
        kinds = [lineage[lineage_depth] for lineage in cluster_lineages]
        diversity = set(kinds)

        enriched_kind, result, kind_ratio = tax.find_enriched_classification(cluster_lineages, phage_lineages, lineage_depth)
        if enriched_kind:
            plt.text(-1, -1.3, "%.1f%% %s p=%.2g" % (100 * kind_ratio, enriched_kind, result[1]))

        ratios, labels, colors = [], [], []
        for kind in diversity:
            ratios.append(kinds.count(kind))
            labels.append(kind.replace('like', ' like '))
            colors.append(color_map[kind])
        colors = np.array(colors)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        patches, texts = plt.pie(ratios, colors=colors, shadow=False, startangle=0, labeldistance=1.0)
        plt.title(titles[lineage_depth - 1])
        plt.legend(patches, labels=labels, fontsize=5, loc='center left', bbox_to_anchor=([0.9, -0.4][lineage_depth % 3 != 0], 0.5), title="Legend", fancybox=True)

        for text in texts:
            text.set_fontsize(5)

    plt.text(1, 1.60, 'ID: %d' % id, fontsize=10)
    plt.text(1, 1.45, 'Category: %d - %s' % (category, category_name(category)), fontsize=10)
    #plt.text(1, 1.30, 'Phamer score: %.3f' % score, fontsize=10)

    ax = plt.subplot(2, 3, 6)
    cluster_silhouettes = phamer.cluster_silhouettes(appended_data, asmt, asmt[-1])
    point_sil = cluster_silhouettes[-1]
    cluster_silhouettes = cluster_silhouettes[:-1]

    plt.barh(0, point_sil, color='red', alpha=0.9)
    plt.barh(range(1, len(cluster_silhouettes)+1), sorted(cluster_silhouettes), color='blue', alpha=0.3)
    plt.xlabel('Silhouette')
    plt.title('Cluster Silhouette')

    plt.savefig(out_file)


def cluster_taxonomy_pies(phamer_summary, virsorter_summary, phage_lineage_file, phage_kmer_file,
                          contig_kmer_file, destination):
    '''
    This function takes True Positive prediction from phamer (as compared to VirSorter) makes plots of the
    lineages of nearby point for each, putting thoes files all into one destination directory with filenames that
    indicate which point is being represented
    :param phamer_summary: The summary file from a Phamer run
    :param virsorter_summary: The summmary file from VirSorter for the same dataset
    :param phage_lineage_file: The lineage file that maps id to lineage for all the phages used
    :param phage_kmer_file: CSV file that contains the k-mer counts of the phage with the GenBank ID in the first column
    :param contig_kmer_file: A CSV file that contains the k-mer counts of the unknown contigs with each contig ID in
    the first column
    :param destination: A directory to put all of the images into
    :return: None
    '''
    ids, phage_kmers = kmer.read_kmer_file(phage_kmer_file, normalize=True)
    lineage_dict = tax.read_lineage_file(phage_lineage_file)
    lineages = [lineage_dict[id] for id in ids]
    lineages = tax.extend_lineages(lineages)

    phamer_dict = phamer.read_phamer_output(phamer_summary)
    tp, fp, fn, tn = get_truth_table(phamer_summary, virsorter_summary)

    color_map = lineage_colors(lineages)

    contig_ids, contig_kmers = kmer.read_kmer_file(contig_kmer_file, normalize=True)
    contig_dict = {int(contig_ids[i]):contig_kmers[i] for i in xrange(len(contig_ids))}
    category_dict = get_category_dictionary(virsorter_summary)
    i = 0
    for id in tp + fn:
        point = contig_dict[id]
        out_file = os.path.join(destination, 'near_lineages_ID_%d.svg' % id)
        score = phamer_dict[id]
        logger.info("Making plot for ID_%d (%s)..." % (id, os.path.basename(out_file)))
        make_lineage_plot(point, score, phage_kmers, lineages, out_file=out_file, color_map=color_map,
                          category=category_dict[id], id=id)
        i += 1
        logger.info("Done. %d of %d" % (i, len(tp)))


def generate_summary(args):
    '''
    This makes a summary for the output file for combining methods
    :param args: The parsed argument parser from function call
    :return: a beautiful summary
    '''
    return args.__str__().replace('Namespace(', '# ').replace(')', '').replace(', ', '\n# ').replace('=', ':\t') + '\n'


def tsne_plot(phamer_summary, virsorter_summary, tsne_file, threshold=0, file_name='comparison_tsne.svg'):
    '''
    This makes a plot that compares results from Phamer and VirSorter by displaying a t-SNE plot
    :param phamer_summary: The output summary file from Phamer
    :param virsorter_summary: The output predicted phage file from VirSorter
    :param tsne_file: A t-SNE file the contains the representation of the points in 2 dimensions
    :param threshold: The threshold to count a Phamer score as a prediction as
    :return: None... just makes a cool plot
    '''
    phamer_dict = phamer.read_phamer_output(phamer_summary)
    true_positives, false_positives, false_negatives, true_negatives = get_truth_table(phamer_summary,virsorter_summary,threshold=threshold)

    ids, tsne_data, chops = phamer.read_tsne_file(tsne_file)
    contig_tsne_ids, phage_tsne_ids, bacteria_tsne_ids = phamer.chop(ids, chops)
    contig_tsne, phage_tsne, bacteria_tsne = phamer.chop(tsne_data, chops)

    plot_dict = {contig_tsne_ids[i]:contig_tsne[i] for i in xrange(len(contig_tsne_ids))}

    TP_points = np.array([plot_dict[id] for id in true_positives])
    FP_points = np.array([plot_dict[id] for id in false_positives])
    FN_points = np.array([plot_dict[id] for id in false_negatives])
    TN_points = np.array([plot_dict[id] for id in true_negatives])

    alpha = 0.9

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.scatter(bacteria_tsne[:, 0], bacteria_tsne[:, 1], s=0.5, c='black', edgecolor='black', alpha=alpha, label='Bacteria')
    plt.scatter(phage_tsne[:, 0], phage_tsne[:, 1], s=0.5, c='blue', edgecolor='blue', alpha=alpha, label='Phage')
    plt.scatter(TN_points[:, 0], TN_points[:, 1], s=3, c='green', edgecolor='green', alpha=alpha,label='True Negatives')
    plt.scatter(FP_points[:, 0], FP_points[:, 1], s=3, c='magenta', edgecolor='magenta', alpha=alpha,label='False Positives')
    plt.scatter(FN_points[:, 0], FN_points[:, 1], s=15, c='red', edgecolor='black', alpha=alpha, label='False Negatives')
    plt.scatter(TP_points[:, 0], TP_points[:, 1], s=15, c='yellow', edgecolor='black', alpha=alpha, label='True Positives')
    for id in ids:
        score = phamer_dict[id]
        if id in true_positives or id in false_negatives or score > 1.5:
            x, y = tsne_data[ids.index(id)]
            #plt.text(x, y, str(id), fontsize=5)

    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(0.96, 0.5), title="Legend", fancybox=True)
    plt.grid(True)
    plt.title('t-SNE comparison of Phamer with VirSorter')
    plt.savefig(file_name)


def performance_ROC(phamer_summary, virsorter_summary, threshold=0, destination=''):
    '''
    This function makes an ROC curve to compare the preformance between Phamer and VirSorter
    :param phamer_summary: The phamer summary file
    :param virsorter_summary: The VirSorter summary fie
    :param threshold: The threshold above which a Phamer score will count as a putative phage prediction
    :param filename: the name of the file to save the ROC image to
    :return: None
    '''
    # Roc Curve
    tp, fp, fn, tn = get_truth_table(phamer_summary, virsorter_summary, threshold=threshold)
    phamer_dict = phamer.read_phamer_output(phamer_summary)
    positive_scores = [phamer_dict[id] for id in tp + fn]
    negative_scores = [phamer_dict[id] for id in fp + tn]
    cv.make_ROC(positive_scores, negative_scores, os.path.join(destination, 'compare_roc.svg'))

    # Box Plot
    category_dict = get_category_dictionary(virsorter_summary)
    phamer_dict = phamer.read_phamer_output(phamer_summary)
    plt.figure()
    data = [[], [], [], []]
    for id in phamer_dict.keys():
        try:
            category = category_dict[id]
        except KeyError:
            category = 0
        score = phamer_dict[id]
        x = 4 - [4, category - (category > 3) * 3][bool(category)]
        data[x].append(score)

    plt.boxplot(data)
    plt.xlabel('Confidence')
    plt.ylabel('Phamer Score')
    plt.savefig(os.path.join(destination, 'category_vs_phamer.svg'))


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    Created by Trent Mick on Fri, 19 Feb 2010 (MIT)
    Taken from: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def preditcion_summary(ids, category_dict, phamer_dict, threshold=0):
    '''
    This function makes a summary for all the phage which were predicted
    :param ids: A list of ids to consider
    :param category_dict: A dictionary mapping id to VirSorter category
    :param phamer_dict: A dictionary mapping id to Phamer score
    :param threshold: Threshold for Phamer score to count as a prediction
    :return: A summary of all the phage which were predicted
    '''

    counts = [0] * 7
    phamer_counts = [0] * 7
    for id in ids:
        try:
            category = category_dict[id]
        except:
            category = 0

        try:
            score = phamer_dict[id]
        except:
            score = -1

        counts[category] += 1
        if score >= threshold:
            phamer_counts[category] += 1

    summary = "Total count: %d\n" % (np.sum(np.array(counts)[[1, 2, 4, 5]]) + np.sum(np.array(phamer_counts)[[3, 5]]))
    for category in xrange(1, 7):
        summary += "VirSorter category %d: %d\t(Phamer: %d)\n" % (category, counts[category], phamer_counts[category])
    return summary.strip()


def make_overview_csv(header_map, phamer_dict, virsorter_dict, img_dict, output_file='final_summary.csv', dataset=None):
    '''
    This function makes a CSV summary of all the results from VirSroter, Phamer, and IMG
    :param header_map: A mapping of contig id to contig header
    :param phamer_dict: A mapping of contig id to Phamer score
    :param virsorter_dict: A mapping of contig id to VirSorter category
    :param img_dict: A mapping of contig id to IMG product and phylogenies
    :param output_file: The output file to save the summary CSV
    :return: None
    '''

    fields = ['Contig_Name', 'Contig_ID', 'Length', 'VirSorter_Category', 'Phamer_Score', 'IMG_Products_and_Phylogenies']
    if dataset is not None:
        fields = fields[:2] + ['DataSet'] + fields[2:]
    ids = virsorter_dict.keys()
    df = pd.DataFrame(columns=fields)
    df['Contig_ID'] = ids
    df['Contig_Name'] = [get_contig_name(header_map[id]) for id in ids]
    df['Length'] = [get_contig_length(header_map[id]) for id in ids]
    df['VirSorter_Category'] = [virsorter_dict[id] for id in ids]
    if dataset is not None:
        df['DataSet'] = dataset

    for id in ids:
        try:
            df.Phamer_Score[df.Contig_ID == id] = phamer_dict[id]
        except KeyError:
            pass

        try:
            df.IMG_Products_and_Phylogenies[df.Contig_ID == id] = img_dict[id]
        except KeyError:
            pass
    df.to_csv(output_file, delimiter=', ')


def make_img_map(img_summary):
    '''
    This function creates a map from cotnig ID to IMG product and phylogeny
    :param img_summary: The IMG summary file generated by img_parser.py
    :return: A dictionary that maps contig ID to IMG product and phylogeny
    '''
    map = {}
    i_map = {}
    for line in [line for line in open(img_summary).readlines() if not line.startswith('#')]:
        id = int(line.split(',')[0])
        product, phylo = line.split(',')[-2:]
        if product.strip() in ['Missing_Product', ''] and phylo.strip() in ['Missing_Phylogeny', '']:
            continue
        try:
            i = i_map[id]
            i_map[id] += 1
        except KeyError:
            i_map[id] = 1
            i = 1
        gene_entry = "%d. %s: %s" % (i, product, phylo)
        try:
            value = map[id]
            value += "%s" % gene_entry
            map[id] = value
        except KeyError:
            map[id] = gene_entry
    return map


if __name__ == '__main__':

    script_description='This script performs secondary analysis on Phamer, VirSorter and IMG data'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-p', '--phamer_summary', type=str, default='', help='Phamer summary file')
    input_group.add_argument('-vs', '--virsorter', type=str, default='', help='Virsorter output directory')
    input_group.add_argument('-t', '--tsne_file', type=str, default='', help='tsne_output_file')
    input_group.add_argument('-pk', '--phage_kmers', type=str, default='', help='Phage kmer file')
    input_group.add_argument('-ck', '--contig_kmers', type=str, default='', help='File of contig k-mer counts, properly formatted')
    input_group.add_argument('-c', '--contigs_file', type=str, default='', help='Original contigs file')
    input_group.add_argument('-img', '--img_directory', type=str, default='', help='IMG directory of gene prediction files')
    input_group.add_argument('-l', '--lineage_file', type=str, default='', help='Phage lineage file')
    input_group.add_argument('-ds', '--dataset', type=str, help='Name of the dataset')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output', type=str, default='', help='Output directory path')

    parser.add_argument('-e', '--email', type=str, default=__email__, help='Email reference for Entrez')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')
    args = parser.parse_args()

    phamer_summary = os.path.expanduser(args.phamer_summary)
    virsorter_out = os.path.expanduser(args.virsorter)
    img_dir = os.path.expanduser(args.img_directory)
    tsne_file = os.path.expanduser(args.tsne_file)
    lineage_file = os.path.expanduser(args.lineage_file)
    phage_kmer_file = os.path.expanduser(args.phage_kmers)
    contig_kmer_file = os.path.expanduser(args.contig_kmers)
    output = os.path.expanduser(args.output)
    contigs_file = os.path.expanduser(args.contigs_file)
    dataset = args.dataset

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')


    # t-SNE plot of all 3 data types
    virsorter_summary = os.path.join(virsorter_out, 'VIRSorter_global-phage-signal.csv')
    if tsne_file and os.path.isfile(tsne_file):
        tsne_plot_file = os.path.join(output, 'comparison_tsne.svg')
        tsne_plot(phamer_summary, virsorter_summary, tsne_file, file_name=tsne_plot_file)

    # ROC curve comparing Phamer and VirSroter
    performance_ROC(phamer_summary, virsorter_summary, threshold=0, destination=output)

    gb_dir = os.path.join(virsorter_out, 'Predicted_viral_sequences')
    gb_files = [os.path.join(gb_dir, filename) for filename in os.listdir(gb_dir) if filename.endswith('.gb')]

    if not os.path.exists(output):
        os.makedirs(output)


    # Preparing files to be viewed with SnapGene
    for file in gb_files:
        logger.info('Preparing %s for SnapGene...' % os.path.basename(file))
        prepare_for_SnapGene(file, os.path.join(output, 'gb_files'))

    # Analysis of IMG files
    if os.path.isdir(img_dir):
        img_summary = os.path.join(output, '%s.summary' % os.path.basename(os.path.relpath(img_dir)))
        logger.info("Analyzing IMG files from %s... Summary file: %s" % (os.path.basename(os.path.relpath(img_dir)), os.path.basename(img_summary)))
        tp, fp, fn, tn = get_truth_table(phamer_summary, virsorter_summary)
        contig_ids = tp + fn
        contig_name_map = get_contig_name_map(contigs_file=contigs_file)
        img.make_gene_csv(img_dir, img_summary, contig_name_map, contig_ids=contig_ids, keyword=None)
    else:
        logger.error("Pass a directory to the -img argument that contains a COG, phylogeny, and product file")

    # Generating a prediction summary
    ids, phage_kmers = kmer.read_kmer_file(phage_kmer_file, normalize=True)
    phamer_dict = phamer.read_phamer_output(phamer_summary)
    tp, fp, fn, tn = get_truth_table(phamer_summary, virsorter_summary)
    contig_ids, contig_kmers = kmer.read_kmer_file(contig_kmer_file, normalize=True)
    contig_dict = {int(contig_ids[i]):contig_kmers[i] for i in xrange(len(contig_ids))}
    category_dict = get_category_dictionary(virsorter_summary)

    header_dict = map_id_to_header(contigs_file)
    virsroter_map = {phage.id: phage.category for phage in read_virsorter_file(virsorter_summary)}
    img_map = make_img_map(img_summary)
    make_overview_csv(header_dict, phamer_dict, virsroter_map, img_map, output_file=os.path.join(output, 'final_summary.csv'), dataset=dataset)

    logger.info("--> Summary of Phamer results:")
    logger.info("True Positives: %d" % len(tp))
    logger.info("False Positives: %d" % len(fp))
    logger.info("False Negatives: %d" % len(fn))
    logger.info("True Negatives: %d" % len(tn))
    logger.info("PPV: %.2f%%" % (100.0 * float(len(tp)) / (len(tp) + len(fp))))
    logger.info(preditcion_summary(map(int, contig_ids), category_dict, phamer_dict))

    # Pie charts representing taxonomy of clusters with contigs
    if query_yes_no('Make new taxonomy pie charts (long computation)?', default="no"):
        logger.info("Making taxonomy pie charts...")
        cluster_taxonomy_pies(phamer_summary, virsorter_summary, lineage_file, phage_kmer_file, contig_kmer_file,
                          os.path.join(output, 'pie_charts'))
    else:
        logger.info("Post analysis complete.")