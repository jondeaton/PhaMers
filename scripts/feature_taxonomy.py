#!/usr/bin/env python
"""
This script makes pie charts of the distributions of taxonomic classifications of sequence data, as well as a t-SNE plot
based on k-mer frequencies, and bar charts of cluster composition based on k-mers and taxonomic classification
"""

import os
import logging
import numpy as np
import taxonomy as tax
import distinguishable_colors as dc
import kmer
import fileIO
import argparse
import learning
import warnings
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
import pylab
from sklearn.manifold import TSNE

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class plot_maker(object):
    
    def __init__(self):
        self.output_directory = 'taxonomy'

        self.features_file = self.get_features_filename()
        self.tsne_file = self.get_tsne_filename()
        self.tsne_plot_filename = self.get_tsne_plot_filename()

        self.do_tsne = False
        self.perplexity = 30
        self.min_samples = 10
        self.cluster_on_tsne = True
        self.dbscan = False
        self.kmeans = True
        self.k_clusters = 40

        if self.cluster_on_tsne:
            self.eps = 5
        else:
            self.eps = 0.014

        self.titles = ['Viruses', 'Baltimore Classification', 'Order', 'Family', 'Subfamily']
        self.markers = ['v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd']

        self.annotate_kinds = True
        self.tsne_figsize = (14, 11)
        self.bar_figsize = (12, 6)

        #todo: figure out what this does when set to True
        self.unknown_toggle = False


    def make_all_plots(self):
        """
        This function makes all the plots
        :return: None
        """

        logger.info("Clustering points...")
        self.assignment = self.get_assignment()


        logger.info("Making pie plots of all data...")
        self.make_pie_charts()
        logger.info("Making t-SNE plot...")
        self.make_tsne_plot()
        logger.info("Making cluster bar charts...")
        self.make_cluster_bar_charts()
        logger.info("Completed all plots.")

    def load_data_for_plotting(self):
        """
        This function gathers all the data necessary for subsequent plotting
        :return: None
        """
        if self.output_directory and not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)

        if not self.do_tsne and os.path.isfile(self.tsne_file) and os.path.isfile(self.feature_file):
            logger.info("Loading t-SNE data from: %s ... " % os.path.basename(self.tsne_file))
            self.id_list, self.tsne_data, _ = fileIO.read_tsne_file(self.tsne_file)
            logger.info("Loaded t-SNE data.")

        elif self.do_tsne:
            if self.feature_file and os.path.exists(self.feature_file):
                logger.info("Loading features from: %s ..." % os.path.basename(self.feature_file))
                self.id_list, self.features = fileIO.read_feature_file(self.feature_file)
                logger.info("Loaded features.")

            elif self.fasta_file and os.path.exists(self.fasta_file):
                logger.info("No feature file provided, calculating features...")
                self.id_list, self.features = kmer.count_file(self.fasta_file, 4, normalize=True)
                logger.info("Calculated features. Saving features to: %s" % os.path.basename(self.features_out))
                kmer.save_counts(self.features, self.id_list, self.features_out, args=args)
                logger.info("Saved features.")

            logger.info("Performing t-SNE...")
            self.tsne_data = TSNE(perplexity=self.perplexity, verbose=True).fit_transform(self.features)
            logger.info("t-SNE complete.")
            self.tsne_file = self.get_tsne_filename()
            fileIO.save_tsne_data(self.tsne_file, self.tsne_data, self.id_list)
            logger.info("Saved t-SNE to: %s" % os.path.basename(self.tsne_file))
        else:
            logger.error("Insufficient input data. Do one of the following:")
            logger.error("1) Supply a CSV file containing t-SNE data with the flag -tsne")
            logger.error("2) Supply a feature file  with the -features flag and add the -do_tsne flag")
            exit(1)

        logger.info("Loading lineages from: %s ..." % os.path.basename(self.lineage_file))
        self.lineages = self.get_lineages()

    def make_pie_charts(self):
        """
        This function makes pie charts of the taxonomic lineage classification of ALL data
        :param lineages: A list of list representing all lineages to consider
        :param output_directory: The directory to drop these .s(a)v(a)g(e) files off in
        :return: None... just makes some pretty chill pie charts
        """
        for lineage_depth in [1, 2, 3]:
            kinds = [lineage[lineage_depth] for lineage in self.lineages]
            diversity = set(kinds)
            fracs, labels = [], []
            for kind in diversity:
                fracs.append(kinds.count(kind))
                labels.append(kind)

            plt.figure(figsize=(8.5, 7))
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            self.colors = dc.get_colors(len(diversity))
            patches, texts = plt.pie(fracs, explode=np.linspace(0.0, 0.0, len(diversity)), colors=self.colors, labeldistance=1.0)
            plt.title(self.titles[lineage_depth])
            plt.legend(patches, labels=labels, fontsize=8, loc='center left', bbox_to_anchor=(0.96, 0.5), title="Legend",  fancybox=True)

            for p in patches:
                p.set_linewidth(0.25)
                p.set_edgecolor('white')

            for text in texts:
                text.set_fontsize(6)

            pie_image_file = self.get_pie_filename(lineage_depth)
            plt.savefig(pie_image_file)
            plt.close()

    def make_tsne_plot(self, file_name=None):
        """
        This function makes a t-SNE plot of phage datapoints and colors them based on clustering after t-SNE reduction.
        Also, the plot will circle clusters that are enriched fora  specific taxonomic classification, and will diplay
        the classification that it is enriched for as well as the percentae of the cluster made of
        :return: None... just makes a savage plot
        """
        num_clusters = 1 + max(self.assignment)
        colors = dc.get_colors(num_clusters)
        plt.figure(figsize=self.tsne_figsize)
        plt.clf()
        axes = pylab.axes()
        ax = plt.subplot(111)
        box = ax.get_position()
        centroids = learning.get_centroids(self.tsne_data, self.assignment)
        used = []
        for cluster in set(self.assignment):
            which = np.arange(len(self.assignment))[self.assignment == cluster]
            cluster_points = self.tsne_data[which]
            cluster_lineages = np.array(self.lineages)[which]
            if self.annotate_kinds:
                # This part puts arrows and text next to each cluster on the t-SNE plot
                for depth in [4, 3, 2, 1, 0]:
                    kind, result, ratio = tax.find_enriched_classification(cluster_lineages, self.lineages, depth)
                    if kind and result and ratio:
                        centroid = centroids[cluster]
                        if kind not in used and 'unclassified' not in kind and ratio >= 0.55 and np.linalg.norm(centroid) >= 0:
                            used.append(kind)
                            radius = np.max(learning.distances(centroid, cluster_points))
                            text_loc = centroid * (1 + (radius + 3) / np.linalg.norm(centroid))
                            x = text_loc[0]
                            y = text_loc[1]
                            text_loc = [x, y]
                            kind_text = "%.1f%% %s" % (100 * ratio, kind)
                            arrowprops = dict(facecolor='black', arrowstyle="->", connectionstyle="arc3")
                            ax.annotate(kind_text, xy=centroid, xytext=text_loc, arrowprops=arrowprops, fontsize=12)
                        break
            label = ['unassigned', '%d' % cluster][cluster >= 0]
            color = colors[cluster]
            marker = self.markers[cluster % len(self.markers)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color,  edgecolor=color, marker=marker, label=label, s=3)

        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}, title="Legend", fontsize=7, ncol=2)
        if file_name is None:
            self.tsne_plot_filename = self.get_tsne_plot_filename()
        plt.savefig(self.tsne_plot_filename)
        plt.close()

    def make_cluster_bar_charts(self):
        """
        This function makes a series of bar charts that displays the classification distribution of the lineages within
        the clusters given by the assignment.
        :return: None
        """
        num_clusters = max(self.assignment)
        logger.debug("Number of clusters: %d" % num_clusters)
        for lineage_depth in [1, 2, 3, 4]:
            title = self.titles[lineage_depth]
            logger.debug("Making bar charts for: %s" % title)
            plt.figure(figsize=self.bar_figsize)
            ax = plt.subplot(111)
            diversity = set([lineage[lineage_depth] for lineage in self.lineages])
            # Making colors to represent each taxa at this depth
            colors = dc.get_colors(len(diversity))
            kind_colors = {list(diversity)[i]: colors[i] for i in xrange(len(diversity))}
            y_offset = np.zeros(num_clusters)
            for kind in diversity:
                fracs = []
                for cluster in xrange(num_clusters):
                    cluster_members = np.arange(len(self.assignment))[self.assignment == cluster]
                    cluster_lineages = np.array(self.lineages)[cluster_members]
                    num_members_of_kind = [lineage[lineage_depth] for lineage in cluster_lineages].count(kind)
                    if self.unknown_toggle:
                        num_cluster_members = len(cluster_members)
                        frac = float(num_members_of_kind) / num_cluster_members
                    else:
                        frac = float(num_members_of_kind)
                    fracs.append(frac)

                color = kind_colors[kind]
                plt.bar(np.arange(num_clusters), fracs, bottom=y_offset, color=color, label=kind, edgecolor=color)
                y_offset += fracs

            plt.xlabel('Cluster')
            plt.title(title)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            prop = {'size': 6}
            ncol = [1, 2][len(diversity) > 50]
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=prop, ncol=ncol, title=title)
            file_name = self.get_barchart_file_name(lineage_depth)
            plt.savefig(file_name)
            plt.close()

    def get_lineages(self):
        """
        This function is for getting lineages out of the lineage file
        :return: A list of lineages
        """
        lineage_dict = fileIO.read_lineage_file(self.lineage_file)
        if self.id_list:
            lineages = [lineage_dict[id] for id in self.id_list]
            lineages = tax.extend_lineages(lineages)
            return lineages
        else:
            logger.warning("No id list in plotter object.")
            logger.warning("Loaded lineages in order that they appear in: %s" % os.path.basename(self.lineage_file))
            return lineage_dict.values()

    def get_assignment(self):
        """
        This function is for getting the cluster assignment of the data
        :return: Whatever data is returned by the dbscan wrapper function in learning.py
        """
        if self.dbscan:
            if self.cluster_on_tsne:
                assignment = learning.dbscan(self.tsne_data, eps=self.eps, min_samples=self.min_samples)
            else:
                assignment = learning.dbscan(self.features, eps=self.eps, min_samples=self.min_samples)
        elif self.kmeans:
            if self.cluster_on_tsne:
                assignment = learning.kmeans(self.tsne_data, self.k_clusters)
            else:
                assignment = learning.dbscan(self.features, self.k_clusters)
        return assignment

    # filename getters
    def get_barchart_file_name(self, lineage_depth):
        """
        This function makes a filename for a cluster barchart
        :param lineage_depth:
        :return:
        """
        lineage_name = self.titles[lineage_depth].lower().split()[0]
        file_name = 'cluster_homology_{lineage_name}_.svg'.format(lineage_name=lineage_name)
        file_name = os.path.join(self.output_directory, file_name)
        return file_name

    def get_pie_filename(self, lineage_depth):
        """
        This function makes a filename for a pie-chart
        :param lineage_depth:
        :return:
        """
        lineage_name = self.titles[lineage_depth].lower().split()[0]
        file_name = 'phage_pie_{lineage_name}.svg'.format(lineage_name=lineage_name)
        file_name = os.path.join(self.output_directory, file_name)
        return file_name

    def get_tsne_filename(self):
        """
        This function makes a file path to a t-SNE data file within the output directoty
        :return: A file path
        """
        return os.path.join(self.output_directory, "tsne_data.csv")

    def get_tsne_plot_filename(self):

        return os.path.join(self.output_directory, "tsne_plot.svg")

    def get_features_filename(self):

        return os.path.join(self.output_directory, "features.csv")


def decide_files(plotter, args):
    """
    This function decides which files to store in the plotter object from the argparse object
    :param plotter: A plotter object
    :param args: An argparse parsed arguments object from this script
    :return: None
    """

    if args.output_directory:
        plotter.output_directory = args.output_directory
        plotter.tsne_plot_filename = plotter.get_tsne_plot_filename()
        plotter.tsne_file = plotter.get_tsne_plot_filename()

    if args.fasta_file:
        plotter.fasta_file = args.fasta_file
    if args.lineage_file:
        plotter.lineage_file = args.lineage_file
    if args.feature_file:
        plotter.feature_file = args.feature_file
    if args.tsne_file:
        plotter.tsne_file = args.tsne_file


if __name__ == '__main__':

    script_description = 'This script makes t-SNE plots relating sequence features to taxonomy'
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-fasta', '--fasta_file',  help='Fasta file')
    input_group.add_argument('-lin', '--lineage_file', help='Lineage file')
    input_group.add_argument('-features', '--feature_file', help='CSV file of features')
    input_group.add_argument('-tsne', '--tsne_file', help='t-SNE file')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_directory', help='Output directory')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-do_tsne', action='store_true', help='Flag to do a new t-SNE run')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')
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

    plotter = plot_maker()
    logger.debug("Storing files paths from argparse in plotter...")
    decide_files(plotter, args)
    plotter.do_tsne = args.do_tsne
    logger.info("Loading data for plotting...")
    plotter.load_data_for_plotting()

    logger.debug("Initiating making all plots...")
    plotter.make_all_plots()

