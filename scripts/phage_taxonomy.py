#!/usr/bin/env python
'''
This script makes pie charts of the distributions of taxonomic classifications of phage, as well as a t-SNE plot
based on k-mer frequencies, and bar charts of cluster composition based on k-mers and taxonomic classification
'''

import os
import sys
import time
import logging
import numpy as np
import taxonomy as tax
import distinguishable_colors as dc
import kmer
import tsne
import argparse
import phamer
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
import pylab

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def make_pie_plots(lineages, out_dir):
    '''
    This function makes pie charts of the taxonomic lineage classification of all phage
    :param lineages: A list of list representing all lineages to consider
    :param out_dir: The directory to drop these .s(a)v(a)g(e) files off in
    :return: None... just makes some pretty chill pie charts
    '''
    titles = ['Baltimore Classification', 'Order', 'Family', 'Sub-Family']
    for lineage_depth in [1, 2, 3]:
        kinds = [lineage[lineage_depth] for lineage in lineages]
        diversity = set(kinds)
        fracs, labels = [], []
        for kind in diversity:
            fracs.append(kinds.count(kind))
            labels.append(kind)

        plt.figure(figsize=(8.5, 7))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        colors = dc.get_colors(len(diversity))
        patches, texts = plt.pie(fracs, explode=np.linspace(0.0, 0.0, len(diversity)), colors=colors, labeldistance=1.0)
        plt.title(titles[lineage_depth - 1])
        plt.legend(patches, labels=labels, fontsize=8, loc='center left', bbox_to_anchor=(0.96, 0.5), title="Legend", fancybox=True)

        for p in patches:
            p.set_linewidth(0.25)
            p.set_edgecolor('white')

        for text in texts:
            text.set_fontsize(6)

        pie_image_file = os.path.join(out_dir, 'phage_pie_%s.svg' % titles[lineage_depth-1].lower().split()[0])
        plt.savefig(pie_image_file)


def phage_tsne_plot(tsne_data, assignment, lineages):
    '''
    This function makes a t-SNE plot of phage datapoints and colors them based on clustering after t-SNE reduction.
    Also, the plot will circle clusters that are enriched fora  specific taxonomic classification, and will diplay
    the classification that it is enriched for as well as the percentae of the cluster made of
    :param tsne_data: The t-SNE data for the phage points
    :param assignment: The cluster assignment of each point in the t-SNE data
    :param lineages: A taxonomic classification list of all the points in the t-SNE data
    :return: None... just makes a savage plot
    '''
    markers = ('v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd')
    num_clusters = max(assignment)
    colors = dc.get_colors(num_clusters)
    plt.figure(figsize=(5, 4))
    axes = pylab.axes()
    ax = plt.subplot(111)
    box = ax.get_position()
    centroids = phamer.get_centroids(tsne_data, assignment)
    used = []
    for cluster in xrange(-1, num_clusters):
        which = np.arange(len(assignment))[assignment == cluster]
        cluster_points = tsne_data[which]
        cluster_lineages = np.array(lineages)[which]
        for depth in [4, 3, 2, 1, 0]:
            kind, result, ratio = tax.find_enriched_classification(cluster_lineages, lineages, depth)
            if kind and result and ratio:
                centroid = centroids[cluster]
                if kind not in used and 'unclassified' not in kind and ratio >= 0.55 and np.linalg.norm(centroid) >= 0:
                    used.append(kind)
                    radius = np.max(phamer.distances(centroid, cluster_points))
                    text_loc = centroid * (1 + (radius + 3) / np.linalg.norm(centroid))
                    kind_text = "%.1f%% %s" % (100 * ratio, kind.replace('like', ' like '))
                    ax.annotate(kind_text, xy=centroid, xytext=text_loc, arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3"), fontsize=7)
                    #plt.text(x, y, kind_text, fontsize=7)
                break
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[cluster],  edgecolor=colors[cluster],marker=markers[cluster % len(markers)], label=('unassigned', '%d' % cluster)[cluster >= 0], s=0.5)
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}, title="Legend", fontsize=7, ncol=2)
    plt.savefig(os.path.join(out_dir, 'tsne_phage_plot.svg'))


def cluster_bar_charts(lineages, assignment):
    '''
    This function makes a series of bar charts that displays the classification distribution of the lineages within
    the clusters given by the assignment.
    :param lineages: A list of taxonomic lineages
    :param assignment: A list of cluster assignments for each lineage
    :return: None
    '''
    titles = ['Baltimore Classification', 'Order', 'Family', 'Subfamily']
    num_clusters = max(assignment)
    for lineage_depth in [1, 2, 3, 4]:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        diversity = set([lineage[lineage_depth] for lineage in lineages])
        colors = dc.get_colors(len(diversity))
        kind_colors = {list(diversity)[i]: colors[i] for i in xrange(len(diversity))}
        y_offset = np.zeros(num_clusters)
        for kind in diversity:
            fracs = []
            for cluster in xrange(num_clusters):
                cluster_members = np.arange(len(assignment))[assignment == cluster]
                cluster_lineages = np.array(lineages)[cluster_members]
                num_members_of_kind = [lineage[lineage_depth] for lineage in cluster_lineages].count(kind)
                # num_cluster_members = len(cluster_members)
                # frac = float(num_members_of_kind) / num_cluster_members
                frac = float(num_members_of_kind)
                fracs.append(frac)
            plt.bar(np.arange(num_clusters), fracs, bottom=y_offset, color=kind_colors[kind],
                    label=kind.replace('like', ' like '), edgecolor=kind_colors[kind])
            y_offset += fracs
        plt.xlabel('Cluster')

        plt.title(titles[lineage_depth - 1])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}, ncol=[1, 2][len(diversity)>50],
        title=titles[lineage_depth - 1])
        image_filename = 'cluster_homology_%s.svg' % titles[lineage_depth-1].lower().split()[0]
        plt.savefig(os.path.join(out_dir, image_filename))


if __name__ == '__main__':

    script_description = 'Make t-SNE pltos of phage taxonomy'
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-f', '--fasta_file', type=str, help='Phage fasta file')
    input_group.add_argument('-l', '--lineage_file', type=str, help='Phage lineage file')
    input_group.add_argument('-pk', '--kmer_file', type=str, help='Phage kmer file')
    input_group.add_argument('-t', '--tsne_file', type=str, help='tsne file')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--out', type=str, help='Output location')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-tsne', action='store_true', help='Flag to do a new t-SNE run')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')
    args = parser.parse_args()

    fasta_file = args.fasta_file
    lineage_file = args.lineage_file
    kmer_file = args.kmer_file
    tsne_file = args.tsne_file
    out_dir = args.out
    do_tsne = args.tsne

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    kmer_out = os.path.join(out_dir, 'phage_kmers.csv')
    tsne_out = os.path.join(out_dir, 'phage_tsne.csv')

    if not do_tsne and os.path.isfile(tsne_file) and os.path.isfile(kmer_file):
        tsne_data = np.loadtxt(tsne_file, delimiter=',')
        phage_ids, kmer_counts = kmer.read_kmer_file(kmer_file, normalize=True)
        logger.info('Loaded tsne data from %s' % os.path.basename(tsne_file))
    elif do_tsne:
        logger.info("Counting k-mers... ")
        tic = time.time()
        phage_ids, kmer_counts = kmer.count_file(fasta_file, 4, normalize=True)
        kmer.save_counts(kmer_counts, phage_ids, kmer_out, args=args)
        logger.info("done. %d minutes, %.2f seconds" % ((time.time() - tic) // 60, (time.time() - tic) % 60))
        # t-SNE
        tic = time.time()
        perplexity = 30
        tsne_data = tsne.tsne(kmer_counts, no_dims=2, perplexity=perplexity)
        logger.info("t-SNE for %d took %.1f seconds" % (len(tsne_data), time.time() - tic))
        np.savetxt(tsne_out, tsne_data, delimiter=',', header='Phage t-sne output file, perplexity=%1.f' % perplexity)
    else:
        logger.error('Not enough data. Do one of the following:')
        logger.error('1) Supply a CSV file of t-SNE data with a headers file by the flags -t and -ph')
        logger.error('2) Supply a kmer count file (-pk) and add the flag -tsne to script call')
        exit(0)

    logger.info("Clustering...")
    # assignment = phamer.dbscan(kmer_counts, eps=0.014, min_samples=10)
    assignment = phamer.dbscan(tsne_data, eps=3.5, min_samples=10)

    lineage_dict = tax.read_lineage_file(lineage_file)
    lineages = [lineage_dict[id] for id in phage_ids]
    lineages = tax.extend_lineages(lineages)

    logger.info("Making plots... ")
    tic = time.time()

    #make_pie_plots(lineages, out_dir)
    phage_tsne_plot(tsne_data, assignment, lineages)
    cluster_bar_charts(lineages, assignment)

    logger.info("done: %d min %d sec" % ((time.time() - tic) // 60, (time.time() - tic) % 60))
