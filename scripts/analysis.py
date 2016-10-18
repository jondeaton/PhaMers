#!/usr/bin/env python
"""
analysis.py

This script is for doing analysis of Phamer results and integrating results with VirSroter and IMG outputs
"""

import os
import warnings
import logging
import argparse
import StringIO
from Bio import Entrez
from Bio import SeqIO
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import matplotlib
matplotlib.use('Agg')
try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dna_features_viewer import GraphicRecord

import fileIO
import id_parser
import taxonomy as tax
import img_parser as img
import cross_validate as cv
import distinguishable_colors
import learning
import basic


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
    """
    This class is used to store data about putative phage predicted by VirSorter
    """
    def __init__(self, kind, category, dataset, line):
        self.kind = kind
        self.category = category
        self.dataset = dataset

        self.header = line.split(',')[0]
        self.id = id_parser.get_contig_id(self.header)
        self.contig_name = id_parser.get_contig_name(self.header)
        self.length = id_parser.get_contig_length(self.header)
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


class results_analyzer(object):
    """
    This class integrates data from IMG, VirSorter and Phamer to produce summary files and plots
    """

    def __init__(self):

        self.input_directory = None
        self.dataset_name = 'unnamed'

        self.data_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.lineage_file = None
        self.virsorter_summary = None
        self.IMG_directory = None
        self.img_summary = None
        self.phamer_summary = None
        self.phage_features_file = None
        self.tsne_file = None

        self.output_directory = None
        self.pie_charts_output = None
        self.diagram_output_directory = None

        self.phage_features = None

        self.phamer_score_threshold = 0

        self.lineage_dict = None

        self.fields = ['Contig_Name', 'Contig_ID', 'Length', 'VirSorter_Category', 'Phamer_Score', 'IMG_Products_and_Phylogenies']
        self.phylogeny_titles = ['Viruses', 'Baltimore Classification', 'Order', 'Family', 'Sub-Family', 'Genus']

        self.cluster_algorithm = 'kmeans'
        self.k_clusters = 86
        self.eps = 0.012
        self.min_samples = 5

        # Plotting options
        self.pie_plot_figsize = (18, 8)
        self.tsne_figsize = (10, 8)
        self.label_contigs_on_tsne = False
        self.phage_color = 'blue'
        self.bacteria_color = 'black'
        self.tp_color = 'yellow'
        self.fp_color = 'magenta'
        self.fn_color = 'red'
        self.tn_color = 'green'

    # Plotting Methods
    def make_pie_lineage_charts(self, id):
        """
        This function makes several lineage pie charts contig based on nearby phage points
        :param id: The id of the
        :return: None... just makes a savage plot
        """
        contig_features = self.contig_features[self.contig_ids == id]

        # Append this contig's features onto those of the reference phage, and then cluster them all
        appended_data = np.vstack((self.phage_features, contig_features))
        if self.cluster_algorithm == 'dbscan':
            asmt = learning.dbscan(appended_data, eps=self.eps, min_samples=self.min_samples)
        elif self.cluster_algorithm == 'kmeans':
            asmt = learning.kmeans(appended_data, self.k_clusters)
        else:
            # The default is k-means I guess...
            asmt = learning.kmeans(appended_data, self.k_clusters)

        # Find which cluster the contig was assigned to
        cluster = asmt[-1]
        cluster_phage = np.arange(self.num_reference_phage)[asmt[:-1] == cluster]
        cluster_lineages = [self.lineages[i] for i in cluster_phage]

        plt.figure(figsize=self.pie_plot_figsize)
        for lineage_depth in xrange(1, 6):
            ax = plt.subplot(2, 3, lineage_depth)
            kinds = [lineage[lineage_depth] for lineage in cluster_lineages]
            diversity = set(kinds)

            enriched_kind, result, kind_ratio = tax.find_enriched_classification(cluster_lineages, self.lineages, lineage_depth)
            if enriched_kind:
                plt.text(-1, -1.3, "%.1f%% %s p=%.2g" % (100 * kind_ratio, enriched_kind, result[1]))

            ratios, labels, colors = [], [], []
            for kind in diversity:
                ratios.append(kinds.count(kind))
                labels.append(kind.replace('like', ' like '))
                colors.append(self.lineage_color_map[kind])
            colors = np.array(colors)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            patches, texts = plt.pie(ratios, colors=colors, shadow=False, startangle=0, labeldistance=1.0)
            plt.title(self.phylogeny_titles[lineage_depth])
            legend_loc = ([0.9, -0.4][lineage_depth % 3 != 0], 0.5)
            plt.legend(patches, labels=labels, fontsize=5, loc='center left', bbox_to_anchor=legend_loc, title="Legend", fancybox=True)

            for text in texts:
                text.set_fontsize(5)

        category = self.contig_category_map[id]
        plt.text(1, 1.60, 'ID: {id}'.format(id=id), fontsize=10)
        plt.text(1, 1.45, 'Category: %d - %s' % (category, category_name(category)), fontsize=10)
        plt.text(1, 1.30, 'Phamer score: %.3f' % self.phamer_dict[id], fontsize=10)

        ax = plt.subplot(2, 3, 6)
        cluster_silhouettes = learning.cluster_silhouettes(appended_data, asmt, asmt[-1])
        point_sil = cluster_silhouettes[-1]
        cluster_silhouettes = cluster_silhouettes[:-1]

        plt.barh(0, point_sil, color='red', alpha=0.9)
        plt.barh(range(1, len(cluster_silhouettes)+1), sorted(cluster_silhouettes), color='blue', alpha=0.3)
        plt.xlabel('Silhouette')
        plt.title('Cluster Silhouette')

        image_file = self.get_pie_charts_file_name(id)
        plt.savefig(image_file)

    def make_cluster_taxonomy_pies(self):
        """
        This function takes True Positive prediction from Phamer (as compared to VirSorter) makes plots of the
        lineages of nearby point for each, putting those files all into one destination directory with file names that
        indicate which point is being represented
        :return: None
        """
        true_positives, false_positives, false_negatives, true_negatives = self.truth_table
        plot_number = 1
        which_ids = true_positives + false_negatives
        output_dir = self.get_pie_charts_output_directory()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for id in which_ids:
            logger.info("Making plots for id: {id} ({num} of {total}) ...".format(id=id, num=plot_number, total=len(which_ids)))
            self.make_pie_lineage_charts(id)
            plot_number += 1

    def make_tsne_plot(self):
        """
        This makes a plot that compares results from Phamer and VirSorter by displaying a t-SNE plot
        :return: None... just makes a cool plot
        """
        if not self.tsne_file:
            logger.error("No t-SNE file found. Did not make scatter plot.")
            return

        ids, tsne_data, chops = fileIO.read_tsne_file(self.tsne_file)
        contig_tsne_ids, phage_tsne_ids, bacteria_tsne_ids = basic.chop(ids, chops)
        contig_tsne, phage_tsne, bacteria_tsne = basic.chop(tsne_data, chops)
        plot_dict = {contig_tsne_ids[i]: contig_tsne[i] for i in xrange(len(contig_tsne_ids))}

        true_positives, false_positives, false_negatives, true_negatives = self.truth_table
        TP_points = np.array([plot_dict[id] for id in true_positives])
        FP_points = np.array([plot_dict[id] for id in false_positives])
        FN_points = np.array([plot_dict[id] for id in false_negatives])
        TN_points = np.array([plot_dict[id] for id in true_negatives])

        alpha = 0.9

        plt.figure(figsize=self.tsne_figsize)
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.scatter(bacteria_tsne[:, 0], bacteria_tsne[:, 1], s=5, c=self.bacteria_color, edgecolor=self.bacteria_color, alpha=alpha, label='Bacteria (%s)' % bacteria_tsne.shape[0])
        plt.scatter(phage_tsne[:, 0], phage_tsne[:, 1], s=0.5, c=self.phage_color, edgecolor=self.phage_color, alpha=alpha, label='Phage (%d)' % self.num_reference_phage)
        plt.scatter(TN_points[:, 0], TN_points[:, 1], s=3, c=self.tn_color, edgecolor=self.tn_color, alpha=alpha,label='True Negatives (%d)' % len(TN_points))
        plt.scatter(FP_points[:, 0], FP_points[:, 1], s=3, c=self.fp_color, edgecolor=self.fp_color, alpha=alpha,label='False Positives (%d)' % len(FP_points))
        plt.scatter(FN_points[:, 0], FN_points[:, 1], s=15, c=self.fn_color, edgecolor='black', alpha=alpha, label='False Negatives (%d)' % len(FN_points))
        plt.scatter(TP_points[:, 0], TP_points[:, 1], s=15, c=self.tp_color, edgecolor='black', alpha=alpha, label='True Positives (%d)' % len(TP_points))

        for id in contig_tsne_ids:
            score = self.phamer_dict[id]
            if id in true_positives or id in false_negatives or score > 1:
                x, y = tsne_data[ids.index(id)]
                if self.label_contigs_on_tsne:
                    plt.text(x, y, str(id), fontsize=5)

        plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(0.96, 0.5), title="Legend", fancybox=True)
        plt.grid(True)
        plt.title('t-SNE comparison of Phamer with VirSorter')

        file_name = self.get_tsne_figname()
        plt.savefig(file_name)

    def make_performance_plots(self):
        """
        This function makes an ROC curve to compare the performance between Phamer and VirSorter
        :return: None
        """
        # Roc Curve
        tp, fp, fn, tn = self.truth_table
        plotter = cv.cross_validator()
        plotter.positive_scores = [self.phamer_dict[id] for id in tp + fn]
        plotter.negative_scores = [self.phamer_dict[id] for id in fp + tn]
        plotter.output_directory = self.output_directory
        plotter.plot_ROC()

        # Box Plot
        plt.figure()
        data = [[], [], [], []]

        for id in self.phamer_dict.keys():
            try:
                category = self.contig_category_map[id]
            except KeyError:
                category = 0

            score = self.phamer_dict[id]
            x = 4 - [4, category - (category > 3) * 3][bool(category)]
            data[x].append(score)

        try:
            plt.boxplot(data)
            plt.xlabel('Confidence')
            plt.ylabel('Phamer Score')
            file_name = self.get_boxplot_filename()
            plt.savefig(file_name)
        except:
            logger.error("Could not make boxplot for the run.")

    def make_contig_diagrams(self):
        """
        This function creates many DNA feature diagrams from all the files within the "genbank" directory.
        :return: None.
        """
        output_dir = self.get_diagram_output_directory()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fun_label = lambda ftr: ftr.qualifiers['product']
        fun_color = None
        features_filter = lambda ftr: ftr.type == 'CDS'

        virsorter_genbank_directory = os.path.join(self.virsorter_directory, "Predicted_viral_sequences")
        genbank_files = basic.search_for_file(virsorter_genbank_directory, end='.gb')

        for genbank_file in genbank_files[1:]:
            # this part formats and parses the files correctly...
            f = open(genbank_file, 'r')
            gb_contents = f.read()
            f.close()
            #gb_contents = format_genbank_str_for_viewing(gb_contents)
            handle = StringIO.StringIO(gb_contents)
            parsed = SeqIO.parse(handle, 'genbank')
            record_dict = SeqIO.to_dict(parsed)
            record = record_dict[record_dict.keys()[0]]
            # This parts plots it
            logger.info("Making diagrams (%d) for %s ..." % (len(record_dict), os.path.basename(genbank_file)))
            for record_key in record_dict:
                record = record_dict[record_key]
                id = id_parser.get_id(record.id)
                num_features = len(record.features)
                plt.subplots()
                graphic_record = GraphicRecord.from_biopython_record(record, fun_label=fun_label, fun_color=fun_color, features_filter=features_filter)
                try:
                    graphic_record.plot(fig_width=num_features * 0.66)
                    file_name = self.get_diagram_filename(id)
                    plt.savefig(file_name)
                except:
                    logger.error("Could not make diagram for: %s" % record.id)
                plt.close()

    # Summary files
    def make_prediction_summary(self, args=None):
        """
        This function makes a summary for all the phage which were predicted by VirSorter and Phamer together
        :return: None. It saves this summary to file.
        """
        all_ids = self.phamer_dict.keys()
        category_counts = np.zeros(7)
        phamer_category_counts = np.zeros(7)
        for id in all_ids:
            try:
                category = self.contig_category_map[id]
            except KeyError:
                category = 0

            try:
                score = self.phamer_dict[id]
            except KeyError:
                score = -2

            category_counts[category] += 1
            if score >= self.phamer_score_threshold:
                phamer_category_counts[category] += 1

        summary = basic.generate_summary(args, header="Summary of Phamer results")
        total_count = np.sum(category_counts[[1, 2, 4, 5]]) + np.sum(phamer_category_counts[[3, 6]])
        summary += "\nTotal/Final count: %d\n" % total_count

        for category in xrange(1, 7):
            summary += "VirSorter Category %s: " % category
            summary += "%d (Phamer: %d)\n" % (category_counts[category], phamer_category_counts[category])

        summary += "True Positives: %d\n" % len(self.truth_table[0])
        summary += "False Positives: %d\n" % len(self.truth_table[1])
        summary += "False Negatives: %d\n" % len(self.truth_table[2])
        summary += "True Negatives: %d\n" % len(self.truth_table[3])
        ppv = float(len(self.truth_table[0])) / (len(self.truth_table[0]) + len(self.truth_table[1]))
        summary += "PPV: %.3f %%\n" % (100.0 * ppv)

        file_name = self.get_prediction_metrics_filename()
        with open(file_name, 'w') as f:
            f.write(summary)
            f.close()

        positive_ids = np.append(self.truth_table[0], self.truth_table[2])
        negative_ids = np.append(self.truth_table[1], self.truth_table[3])
        positive_scores = np.array([self.phamer_dict[id] for id in positive_ids])
        negative_scores = np.array([self.phamer_dict[id] for id in negative_ids])
        metrics_series = learning.get_predictor_metrics(positive_scores, negative_scores, threshold=self.phamer_score_threshold)

        metric_name_map = {'tp': "True Positives", "fp": "False Positives", "fn": "False Negtives",
                           "tn": "True Negatives", "tpr": "True Positive Rate", "fpr": "False Positive Rate",
                           "fnr": "False Negative Rate", "tnr": "True Negative Rate",
                           "ppv": "Positive Predictive Value", "npv": "Negative Predictive Value",
                           "fdr": "False Discovery Rate", "acc": "Accuracy"}
        new_series = pd.Series(index=metric_name_map.values())
        for metric in metric_name_map.keys():
            new_series[metric_name_map[metric]] = metrics_series[metric]
        new_series.to_csv(file_name, sep="\t", mode='a')

    def make_overview_csv(self):
        """
        This function makes a CSV summary of all the results from VirSroter, Phamer, and IMG
        :return: None
        """
        if self.dataset_name is not None:
            fields = self.fields[:2] + ['DataSet'] + self.fields[2:]
        ids = self.virsroter_map.keys()
        df = pd.DataFrame(columns=fields)
        df['Contig_ID'] = ids
        df['Contig_Name'] = [id_parser.get_contig_name(self.id_header_map[id]) for id in ids]
        df['Length'] = [id_parser.get_contig_length(self.id_header_map[id]) for id in ids]
        df['VirSorter_Category'] = [self.virsroter_map[id] for id in ids]
        if self.dataset_name is not None:
            df['DataSet'] = self.dataset_name

        for id in ids:
            try:
                df.Phamer_Score[df.Contig_ID == id] = self.phamer_dict[id]
            except KeyError:
                pass

            try:
                df.IMG_Products_and_Phylogenies[df.Contig_ID == id] = self.img_proucts_map[id]
            except KeyError:
                pass

        summary_file_path = self.get_prediction_summary_filename()
        df.to_csv(summary_file_path, delimiter=', ')

    def make_gene_csv(self):
        """
        This function parses files from a directory that contains a IMG COG, phylogeny, and product files and writes the
        compiled data of contig id, phylogenies, and products into a tab-separated summary file
        :return: None
        """
        self.img_summary = self.get_img_summary_filename()
        f = open(self.img_summary, 'w')
        header = img.gene_csv_header(self.img_directory)
        f.write(header)
        for contig_id in self.img_proucts_map:
            next_contig = self.img_proucts_map[contig_id]
            f.write(str(next_contig) + '\n')

    def prepare_gb_files_for_SnapGene(self):
        """
        This function prepares all of the GenBank files in this object for viewing with SnapGene
        :return: None
        """
        output_dir = self.get_genbank_output_directory()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for file in self.genbank_files:
            prepare_for_SnapGene(file, output_dir)

    # Basic Functions
    def load_data(self):
        """
        This function loads necessary data into the analyzer object that will be used for subsequent analyses. It gets
        data from the files who's paths are stored as attributes in side the object.
        :return: None
        """
        logger.debug("Loading ids...")
        self.id_header_map = get_id_header_map(self.fasta_file)
        logger.debug("Loading features...")
        self.contig_ids, self.contig_features = fileIO.read_feature_file(self.features_file, normalize=True)
        logger.debug("Loading VirSorter results...")
        self.contig_category_map = self.get_contig_category_map()
        logger.debug("Loading Phamer results...")
        self.phamer_dict = fileIO.read_phamer_output(self.phamer_summary)

        logger.debug("Loading phage features...")
        self.phage_ids, self.phage_features = fileIO.read_feature_file(self.phage_features_file, normalize=True)
        self.num_reference_phage = len(self.phage_ids)
        logger.debug("Loading lineages...")
        self.lineage_dict = fileIO.read_lineage_file(self.lineage_file, extend=True)
        self.lineages = self.get_lineages()
        self.lineage_color_map = self.get_lineage_colors()

        logger.debug("Getting truth table...")
        self.truth_table = self.get_virsorter_phamer_truth_table()
        self.virsroter_map = {phage.id: phage.category for phage in fileIO.read_virsorter_file(self.virsorter_summary)}

        logger.debug("Loading contig names and ids...")
        self.contig_name_id_nap = get_name_id_map(contigs_file=self.fasta_file)
        logger.debug("Loading IMG data...")
        self.img_proucts_map = img.get_contig_map(self.img_directory, self.contig_name_id_nap)
        logger.info("Loaded all data.")

    def get_lineages(self):
        """
        This function returns lineages
        :return: A list of pre-extended lineages that are in the same order as the phage ids read
        from a features file
        """
        return np.array([self.lineage_dict[id] for id in self.phage_ids])

    def get_lineage_colors(self):
        """
        This function makes a mapping of taxonomic classification to color so that multiple plots can be colored
        consistently
        :return: A map of taxonomic classification to color. The color is a 1x3 numpy array representing a RGB color
        """
        diversity = set()
        for depth in xrange(tax.deepest_classification(self.lineages)):
            diversity |= set([lineage[depth] for lineage in self.lineages])

        diversity = list(diversity)
        colors = distinguishable_colors.get_colors(len(diversity))
        color_map = {diversity[i]:colors[i] for i in xrange(len(diversity))}
        return color_map

    def get_contig_category_map(self):
        """
        This function takes a VirSroter summary file and creates a dictionary that maps contig ID to VirSorter category
        :param virsorter_summary: The VirSorter summary file name as a string
        :return: A dictionary that mapping contig ID to VirSorter category
        """
        lines = open(self.virsorter_summary, 'r').readlines()
        category_map = {}
        category = 0
        for line in lines:
            if line.startswith('##') and 'category' in line:
                category = int(line[3])
            elif not line.startswith('##'):
                id = id_parser.get_contig_id(line.split(',')[0])
                category_map[id] = category
        return category_map

    def get_virsorter_phamer_truth_table(self):
        """
        This function finds the true positives, false positives, false negatives, and true positives for phage
        predictions from phamer and from VirSorter
        :return: A tuple of lists of IDs for true positives, false positives, false negatives, and true positives in that order
        """
        phamer_ids = [id for id in self.phamer_dict.keys() if self.phamer_dict[id] >= self.phamer_score_threshold]

        vs_phage = fileIO.read_virsorter_file(self.virsorter_summary)
        for phage in vs_phage:
            if phage.id in self.phamer_dict.keys():
                phage.phamer_score = self.phamer_dict[phage.id]
        vs_ids = [phage.id for phage in vs_phage]

        true_positives = [id for id in self.phamer_dict.keys() if id in vs_ids and id in phamer_ids]
        false_positives = [id for id in self.phamer_dict.keys() if id in phamer_ids and id not in vs_ids]
        false_negatives = [id for id in self.phamer_dict.keys() if id in vs_ids and id not in phamer_ids]
        true_negatives = [id for id in self.phamer_dict.keys() if id not in phamer_ids and id not in vs_ids]

        logger.info("%d true positives" % len(true_positives))
        logger.info("%d false positives" % len(false_positives))
        logger.info("%d false negatives" % len(false_negatives))
        logger.info("%d true negatives" % len(true_negatives))

        return true_positives, false_positives, false_negatives, true_negatives

    # File Location
    def find_input_files(self):
        """
        This function finds input files within the input directory field of this object based on file names
        within that directory
        :return: None.
        """
        if self.input_directory and os.path.isdir(self.input_directory):
            fasta_files = basic.search_for_file(self.input_directory, end=".fasta")
            if len(fasta_files) == 1:
                self.fasta_file = fasta_files[0]

            phamer_directory = os.path.join(self.input_directory, "phamer_output")
            phamer_files = basic.search_for_file(phamer_directory, contain='scores', end='.csv')
            if len(phamer_files) == 1:
                self.phamer_summary = phamer_files[0]
            tsne_files = basic.search_for_file(phamer_directory, contain='tsne', end='.csv')
            if len(tsne_files) == 1:
                self.tsne_file = tsne_files[0]

            features_files = basic.search_for_file(self.input_directory, end='.csv')
            if len(features_files) == 1:
                self.features_file = features_files[0]

            virsorter_directories = basic.search_for_file(self.input_directory, start='VirSorter')
            if len(virsorter_directories) == 1:
                self.virsorter_directory = virsorter_directories[0]
                self.find_virsorter_files()

            img_directories = basic.search_for_file(self.input_directory, start='IMG_')
            if len(img_directories) == 1:
                self.img_directory = img_directories[0]

    def find_data_files(self):
        """
        This function finds data files within the data directory field of this object based on file names
        within that directory
        :return: None.
        """
        if self.data_directory and os.path.isdir(self.data_directory):
            self.phage_features_file = os.path.join(self.data_directory, "reference_features", "positive_features.csv")
            self.lineage_file = os.path.join(self.data_directory, "phage_lineages.txt")

    def find_genbank_files(self):
        """
        This function finds all of the GenBank files files within th genbank_directory field of this object
        and stores them in a variable called genbank_files
        :return: None
        """
        self.genbank_files = basic.search_for_file(self.genbank_directory, end='.gb')

    def find_virsorter_files(self):
        """
        This function finds the relevant VirSorter outptu files from within the VirSorter directory and stores them
        :return: None
        """
        self.virsorter_summary = os.path.join(self.virsorter_directory, 'VIRSorter_global-phage-signal.csv')
        self.genbank_directory = os.path.join(self.virsorter_directory, 'Predicted_viral_sequences')
        self.find_genbank_files()

    # File name getters
    def get_diagram_output_directory(self):
        """
        This function is for agetting a location to put contig gene diagrams
        :return: Thsi function returns a path to a subdirectory of the output directory
        """
        return os.path.join(self.output_directory, "contig_diagrams")

    def get_diagram_filename(self, id):
        """
        This function is for getting a path to a file to saving a dna diagram
        :return: A path to a file where I could save a DNA diagram
        """
        if not self.diagram_output_directory:
            self.diagram_output_directory = self.get_diagram_output_directory()
        return os.path.join(self.diagram_output_directory, "dna_features_contig_{id}.svg".format(id=id))

    def get_genbank_output_directory(self):
        """
        This function gets a directory for where GenBank files shold be output to
        :return:
        """
        return os.path.join(self.output_directory, "genbank_files")

    def get_pie_charts_output_directory(self):
        """
        This function makes a file path to a directory where all the pie charts should be put
        :return:
        """
        return os.path.join(self.output_directory, 'pie_charts')

    def get_pie_charts_file_name(self, id):
        """
        Returns a file path for a figure with pie charts for the contig
        :param id: The id of the contig
        :return: a file path for a figure with pie charts for the contig
        """
        if not self.pie_charts_output:
            self.pie_charts_output = self.get_pie_charts_output_directory()
        return os.path.join(self.pie_charts_output, 'close_phage_contig_{id}.svg'.format(id=id))

    def get_tsne_figname(self):
        """
        This function returns a filename for a t-SNE image
        :return:
        """
        return os.path.join(self.output_directory, "tsne_comparison.svg")

    def get_boxplot_filename(self):
        """
        This function returns a file path for a boxplot
        :return:
        """
        return os.path.join(self.output_directory, "category_boxplot.svg")

    def get_prediction_metrics_filename(self):
        """
        This function returns a file path for a summary file
        :return: A file path as a string
        """
        return os.path.join(self.output_directory, "prediction_metrics.csv")

    def get_prediction_summary_filename(self):
        """
        This function returns a suitable file path to a place to store the integrated data from VirSorter, Phamer
        and IMG analyses
        :return: A file path as a string
        """
        return os.path.join(self.output_directory, "integrated_summary.csv")

    def get_img_summary_filename(self):
        """
        This function makes a path for an img summary output file
        :return: A absolute path to the summary outptu files
        """
        return os.path.join(self.output_directory, '%s.summary' % os.path.basename(os.path.relpath(self.img_directory)))


# === FUNCTIONS ===
def get_name_id_map(contigs_file=None, headers=None):
    """
    This file creates a dictionary that maps contig name to contig id
    :param headers: A list of contig headers
    :return: A map from contig name to id
    """
    if contigs_file:
        headers = [line for line in open(contigs_file, 'r').readlines() if line.startswith('>')]

    map = {}
    for header in headers:
        name = id_parser.get_contig_name(header)
        id = id_parser.get_contig_id(header)
        map[name] = id
    return map


def get_id_header_map(contigs_file):
    """
    This function makes a mapping from Contig ID to contig header
    :param contigs_file: The contigs file containing all contigs with headers starting with '>'
    :return: A dictionary that maps contig id to contig header
    """
    headers = [line[1:] for line in open(contigs_file, 'r').readlines() if line.startswith('>')]
    map = {id_parser.get_contig_id(header): header for header in headers}
    return map


def category_name(category):
    """
    This function returns the qualitative confidence that a phage of a certain category is a phage
    :param category: The integer from 1 to 3 that represents the category of VirSorter prediction
    :return: The qualitative confidence that a phage of a certain category is a phage
    """
    category_names = ['unknown', 'Complete - "pretty sure"', 'Complete - "quite sure"', 'Complete - "not so sure"']
    category_names += ['Prophages - "pretty sure"', 'Prophages - "quite sure"', 'Prophages -"not so sure"']
    return category_names[category]


# GenBank files
def get_gene_product_dict(genbank_file):
    """
    This function finds and returns a mapping of gene name to gene product name from a GenBank file containing only a
    single gene product
    :param genbank_file: The GenBank file to analyze
    :return: A dictionary that maps gene name to gene product name for all genes in the file
    """
    gene_product_dict = {}
    gen = SeqIO.parse(genbank_file, 'genbank')
    for record in gen:
        for feature in record.features:
            if feature.type == 'CDS':
                name = feature.qualifiers['gene'][0]
                product = feature.qualifiers['product'][0]
                #logger.debug("%s: %s" % (name, product))
                gene_product_dict[name] = product
    return gene_product_dict


def prepare_for_SnapGene(genbank_file, destination):
    """
    This function prepares a GenBank file from VirSorter to be viewed by SnapGene so that the product names are
    displayed instead of the gene names. (i.e. 'hypothetical protein' instead of 'gene_1')
    :param genbank_file: The output file from VirSorter that shows predicted phage in the GenBank format
    :param destination: The filename of the new changed file
    :return: None
    """
    contents = open(genbank_file, 'r').read()
    if not contents == '':
        putative_phage_gb = contents.split('//')
        putative_phage_gb = [ppgb for ppgb in putative_phage_gb if not ppgb.strip() == '']
    else:
        putative_phage_gb = []

    if destination and not os.path.isdir(destination):
        os.mkdir(destination)

    for phage_gb in putative_phage_gb:
        contig_header = phage_gb.strip().split('\n')[0].split()[1]
        new_gb_file = 'VS_SuperContig_%s_ID_%s.gb' % (id_parser.get_contig_name(contig_header), id_parser.get_contig_id(contig_header))
        new_gb_file = os.path.join(destination, new_gb_file)
        f = open(new_gb_file, 'w')
        f.write(phage_gb)
        f.close()
        gene_prod_dict = get_gene_product_dict(new_gb_file)

        f = open(new_gb_file, 'r')
        contents = f.read()
        f.close()

        for gene in gene_prod_dict.keys():
            contents = contents.replace('/gene="%s"' % gene, '/gene="%s"' % gene_prod_dict[gene])

        # This was an attempt at making these files viewable by dna_feature_viewer
        #lines = contents.split("\n")
        #contents = "\n".join([lines[0]] + [line.replace('/', '') for line in lines[1:]])
        #contents.replace("\n\n", "\n")
        contents = contents.strip() + '\n//'

        f = open(new_gb_file, 'w')
        f.write(contents)
        f.close()


def format_genbank_str_for_viewing(genbank_string):
    """
    This function makes a genbank string ready to be displayed with DNA feature viewer
    :param genbank_string: A string of a genbank file
    :return: A formatted string
    """
    parts = genbank_string.split('//')
    for i in xrange(len(parts)):
        part = parts[i]
        lines = part.split("\n")
        lines = lines[:7] + [line.replace('/', '') for line in lines[7:]]
        parts[i] = '\n'.join(lines)
    genbank_string = '//'.join(parts) + '//'
    open("/Users/jonpdeaton/Desktop/formatted.gb", 'w').write(genbank_string)
    return genbank_string


def decide_files(analyzer, args):
    """
    This function decides which files to use comparins defaules and those provided by the user in by argparse and
    loads those file paths into the results_analyzer object
    :param analyzer: A results_analyzer object
    :param args: An argparse parsed arguments object
    :return: None
    """
    # Input files
    if args.input_directory:
        analyzer.input_directory = args.input_directory
        analyzer.find_input_files()

    if args.fasta_file:
        analyzer.fasta_file = args.fasta_file

    if args.features_file:
        analyzer.features_file = args.features_file

    if args.phamer_summary:
        analyzer.phamer_summary = args.phamer_summary

    if args.tsne_file:
        analyzer.tsne_file = args.tsne_file

    if args.img_directory:
        analyzer.img_directory = args.img_directory
        analyzer.find_img_files()

    if args.virsorter_directory:
        analyzer.virsorter_directory = args.virsorter_directory
        analyzer.find_virsorter_files()

    # Data files
    if args.data_directory:
        analyzer.data_directory = args.data_directory
        analyzer.find_data_files()

    if args.lineage_file:
        analyzer.lineage_file = args.lineage_file

    if args.phage_features:
        analyzer.phage_features_file = args.phage_features

    # Outputs
    if args.output_directory:
        analyzer.output_directory = args.output_directory
    elif args.input_directory:
        analyzer.output_directory = os.path.join(args.input_directory, "phamer_output")
    elif args.fasta_file:
        analyzer.output_directory = os.path.join(os.path.dirname(args.fasta_file), "phamer_output")
    elif args.features_file:
        analyzer.output_directory = os.path.join(os.path.dirname(args.features_file), "phamer_output")


    # Deciding what the dataset name should be...
    if args.dataset_name:
        analyzer.dataset_name = args.dataset_name
    elif args.input_directory:
        head, tail = os.path.split(args.input_directory)
        if tail != '':
            analyzer.dataset_name = tail
        else:
            analyzer.dataset_name = os.path.basename(head)


if __name__ == '__main__':

    script_description = 'This script performs secondary analysis on Phamer, VirSorter and IMG data'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input_directory', help="Direcotry containing all input files")
    input_group.add_argument('-fasta', '--fasta_file', help='Original contigs fasta file')
    input_group.add_argument('-features', '--features_file', help='File of contig k-mer counts, properly formatted')
    input_group.add_argument('-phamer', '--phamer_summary', help='Phamer summary file')
    input_group.add_argument('-tsne', '--tsne_file', help='t-SNE file')
    input_group.add_argument('-img', '--img_directory', help='IMG directory of gene prediction files')
    input_group.add_argument('-vs', '--virsorter_directory', help='VirSorter output directory')

    data_group = parser.add_argument_group("Data")
    data_group.add_argument('-data', "--data_directory", help="Directory containing all relevant data files")
    data_group.add_argument('-lin', '--lineage_file', help="Phage lineage file")
    data_group.add_argument('-pf', '--phage_features', help="Phage features file")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_directory', help='Output directory path')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-name', '--dataset_name',  help='Name of the data set')
    options_group.add_argument('-e', '--email', default=__email__, help='Email reference for Entrez')

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

    analyzer = results_analyzer()
    decide_files(analyzer, args)
    logger.info("Loading data from files...")
    analyzer.load_data()

    if not os.path.isdir(analyzer.output_directory):
        os.mkdir(analyzer.output_directory)

    logger.info("Making t-SNE plot...")
    analyzer.make_tsne_plot()
    logger.info("Plotting ROC curve and boxplot...")
    analyzer.make_performance_plots()
    logger.info("Preparing GenBank files for SnapGene...")
    analyzer.prepare_gb_files_for_SnapGene()
    logger.info("Making prediction metrics file...")
    analyzer.make_prediction_summary(args=args)
    logger.info("Making summary file...")
    analyzer.make_overview_csv()
    logger.info("Making gene csv file...")
    analyzer.make_gene_csv()
    logger.info("Making contig diagrams...")
    analyzer.make_contig_diagrams()
    logger.info("Making taxonomy pie charts...")
    analyzer.make_cluster_taxonomy_pies()

    logger.info("Analysis complete.")