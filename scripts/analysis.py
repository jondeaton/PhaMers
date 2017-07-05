#!/usr/bin/env python
"""
analysis.py

This script is for doing analysis of PhaMers results and integrating results with VirSroter and IMG outputs.
This script does a lot of things...
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
matplotlib.use('agg')
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('agg')
warnings.simplefilter('ignore', UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from dna_features_viewer import GraphicRecord

import fileIO
import id_parser
import taxonomy as tax
import img_parser as img
import distinguishable_colors
import learning
import basic

# This is so that the PDF images created have editable text (For Adobe Illustrator)
matplotlib.rcParams['pdf.fonttype'] = 42
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
        self.img_directory = None
        self.img_summary = None
        self.phamer_summary = None
        self.phage_features_file = None
        self.tsne_file = None
        self.fasta_file = None
        self.features_file = None

        self.output_directory = None
        self.pie_charts_output = None
        self.diagram_output_directory = None

        self.taxonomy_prediction_dict = None
        self.cluster_silhouette_map = None
        self.cluster_lineage_map = None

        self.phage_features = None
        self.phamer_score_threshold = 0
        self.strict = False
        self.lineage_dict = None
        self.taxonomy_prediction_dict = None

        self.summary_fields = ['Header', 'Contig_Name', 'Contig_ID', 'Length', 'VirSorter_Category', 'Phamer_Score', 'IMG_Products_and_Phylogenies', "PhaMers_Taxonomy"]
        self.phylogeny_names = ['Viruses', 'Baltimore', 'Order', 'Family', 'Sub-Family', 'Genus']

        self.cluster_algorithm = 'kmeans'
        self.k_clusters = 86
        self.eps = 0.012
        self.min_samples = 5

        # Plotting options
        self.ids_to_diagram = []
        self.have_been_diagramed = []
        self.pie_plot_figsize = (18, 8)
        self.tsne_figsize = (10, 8)
        self.label_contigs_on_tsne = True
        self.ids_to_label = ['15', '2115', '317', '4273', '5193', '5519', '6299', '873'] # Only these will be labeled on t-SNE (empty = label all contigs)
        self.phage_color = 'blue'
        self.bacteria_color = 'black'
        self.tp_color = 'yellow'
        self.fp_color = 'magenta'
        self.fn_color = 'red'
        self.tn_color = 'green'
        self.dot_size = 5

    # Plotting Methods
    def make_all_plots(self):
        """
        This function makes all plots and saves them
        :return: None
        """

        fig, ax = plt.subplots(1, figsize=self.tsne_figsize)
        self.plot_tsne(ax=ax)
        fig.savefig(self.get_tsne_figname())
        plt.close(fig)

        fig, ax = plt.subplots(1)
        self.plot_roc(ax=ax)
        fig.savefig(self.get_roc_filename())
        plt.close(fig)

        try:
            fig, ax = plt.subplots(1)
            self.plot_boxplot(ax=ax)
            fig.savefig(self.get_boxplot_filename())
            plt.close(fig)
        except:
            logger.error("Could not make boxplot.")

        self.diagram_contigs()
        self.make_contig_plots()

    def plot_tsne(self, ax=None):
        """
        This makes a plot that compares results from Phamer and VirSorter by displaying a t-SNE plot
        :return: None... just makes a savage plot
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
        if ax is None:
            fig, ax = plt.subplots(1, figsize=self.tsne_figsize)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.scatter(bacteria_tsne[:, 0], bacteria_tsne[:, 1], s=self.dot_size, c=self.bacteria_color, edgecolor=self.bacteria_color, alpha=alpha, label='Reference Bacteria (%s)' % bacteria_tsne.shape[0])
        ax.scatter(phage_tsne[:, 0], phage_tsne[:, 1], s=self.dot_size, c=self.phage_color, edgecolor=self.phage_color, alpha=alpha, label='Reference Phage (%d)' % self.num_reference_phage)

        if TN_points.shape[0]:
            ax.scatter(TN_points[:, 0], TN_points[:, 1], s=self.dot_size, c=self.tn_color, edgecolor=self.tn_color, alpha=alpha,label='Novel Bacteria (%d)' % len(TN_points))
        if FP_points.shape[0]:
            ax.scatter(FP_points[:, 0], FP_points[:, 1], s=self.dot_size, c=self.fp_color, edgecolor=self.fp_color, alpha=alpha,label='PhaMers Only (%d)' % len(FP_points))
        if FN_points.shape[0]:
            ax.scatter(FN_points[:, 0], FN_points[:, 1], s=2*self.dot_size, c=self.fn_color, edgecolor='black', alpha=alpha, label='VirSorter Only (%d)' % len(FN_points))
        if TP_points.shape[0]:
            ax.scatter(TP_points[:, 0], TP_points[:, 1], s=2*self.dot_size, c=self.tp_color, edgecolor='black', alpha=alpha, label='PhaMers and VirSorter (%d)' % len(TP_points))

        for id in contig_tsne_ids:
            score = self.phamer_dict[id]
            if id in true_positives or id in false_negatives or score > 1:
                x, y = tsne_data[ids.index(id)]
                if self.label_contigs_on_tsne and (not self.ids_to_label or id in self.ids_to_label):
                    ax.text(x, y, str(id), fontsize=5)

        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(0.96, 0.5), title="Legend", fancybox=True)
        ax.grid(True)
        ax.set_title('t-SNE comparison of Phamer with VirSorter')
        return ax

    def plot_roc(self, ax=None):
        """
        This function plots an ROC curve that compares the performacne of PhaMers against VirSorter
        :param ax: A set of axes to plot the ROC curve on
        :return: The axes
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        which_strictness = [False, True]
        colors = ['b', 'r']
        labels = ['All', 'Confident']
        for i in xrange(2):
            tp, fp, fn, tn = self.get_virsorter_phamer_truth_table(strict=which_strictness[i])
            positive_scores = [self.phamer_dict[id] for id in tp + fn]
            negative_scores = [self.phamer_dict[id] for id in fp + tn]
            fpr, tpr, roc_auc = learning.predictor_performance(positive_scores, negative_scores)
            ax.plot(fpr, tpr, colors[i], label='%s (AUC: %0.3f)' % (labels[i], roc_auc))
            if len(tp) > 0 and len(fn) > 0:
                tpr = len(tp) / float(len(tp) + len(fn))
            else:
                tpr = 0
            if len(fp) > 0 and len(tn) > 0:
                fpr = len(fp) / float(len(fp) + len(tn))
            else:
                fpr = 0
            ax.plot(fpr, tpr, 'o%s' % colors[i])

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(b=True, which='minor')
        return ax

    def plot_boxplot(self, ax=None):
        """
        This method makes a box-whisker plot that shows the distributions of PhaMers scores as a function
        of the confidence with which VirSorter classified the contig as viral
        :param ax: A matplotlib axes to put the plot on
        :return: The axes on which the boxplot was plotted
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        data = [[], [], [], []]
        for id in self.phamer_dict.keys():
            try:
                category = self.contig_category_map[id]
            except KeyError:
                category = 0
            score = self.phamer_dict[id]
            x = 4 - [4, category - (category > 3) * 3][bool(category)]
            data[x].append(score)

        ax.boxplot(data)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Phamer Score')
        return ax

    def plot_cluster_silhouettes(self, cluster_silhouettes, ax=None):
        """
        This function makes a horizontal bar chart of the cluster silhouettes
        :param ax: A matplotlib axes to plot onto
        :return: The matplotlib axes that the plot was put onto
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        cluster_size = len(cluster_silhouettes)
        if cluster_size > 0:
            point_sil = cluster_silhouettes[-1]
            cluster_silhouettes = cluster_silhouettes[:-1]
            ax.barh(0, point_sil, color='red', alpha=0.9)
            ax.barh(range(1, len(cluster_silhouettes) + 1), sorted(cluster_silhouettes), color='blue', alpha=0.3)
            ax.set_xlim([-1, 1])
            ax.set_ylim([0, cluster_size])
            ax.set_xlabel("Silhouette", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.set_title("Cluster Silhouettes", fontsize=10)
        return ax

    def plot_cluster_composition(self, cluster_lineages, ax=None):
        """
        This makes a plot of cluster composition
        :param ax: An axes to put the plot onto
        :return: The axes that the bar char was plotted onto
        """
        cluster_size = len(cluster_lineages)
        dims = len(cluster_lineages.shape)
        if dims == 1:
            cluster_lineages = np.array([cluster_lineages])
        lineage_depth = cluster_lineages.shape[1]
        all_kinds = set(cluster_lineages.ravel())
        color_dict = distinguishable_colors.get_color_dict(all_kinds)
        y_offset = np.zeros(lineage_depth)
        x = np.arange(lineage_depth)
        for kind in all_kinds:
            color = color_dict[kind]
            y = np.array([list(cluster_lineages[:, depth]).count(kind) for depth in xrange(lineage_depth)])
            ax.bar(np.arange(cluster_lineages.shape[1]), y, bottom=y_offset, color=color, label=kind, edgecolor=color)
            y_offset += y

        ax.legend(fontsize=5, bbox_to_anchor=(-0.1, 1), ncol=1 + int(len(all_kinds) > 20))
        ax.set_xlim([0, 5])
        ax.set_ylim([0, cluster_size])
        plt.xticks(0.25 + np.arange(5), self.phylogeny_names, rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=9)
        plt.title("Cluster Taxonomy", fontsize=7)
        plt.show()

    def diagram_contigs(self):
        """
        This function creates many DNA feature diagrams from all the files within the "genbank" directory.
        :return: None
        """
        if self.virsorter_directory is None or not os.path.isdir(self.virsorter_directory):
            logger.info("No VirSorter directory to make contig diagrams")
            return

        output_dir = self.get_diagram_output_directory()
        if output_dir and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fun_label = lambda feature: feature.qualifiers['product'][0]
        fun_color = seq_features_to_color
        features_filter = lambda feature: feature.type == 'CDS'

        virsorter_genbank_directory = os.path.join(self.virsorter_directory, "Predicted_viral_sequences")
        genbank_files = basic.search_for_file(virsorter_genbank_directory, end='.gb')

        for genbank_file in genbank_files:
            f = open(genbank_file, 'r')
            gb_contents = f.read()
            f.close()
            record_dict = SeqIO.to_dict(SeqIO.parse(StringIO.StringIO(gb_contents), 'genbank'))
            # CONTIG DIAGRAM
            logger.info("Making %d diagrams for %s..." % (len(record_dict), os.path.basename(genbank_file)))
            for record_key in record_dict:
                record = record_dict[record_key]
                id = id_parser.get_id(record.id)
                if self.ids_to_diagram is not None and id not in self.ids_to_diagram:
                    logger.info("Skipping id: %s" % id)
                    continue
                logger.info("Diagramming: %s..." % id)
                num_features = len(record.features)
                fig_width = max(15, num_features * 0.5)
                dims = [2, max(5, num_features / 6)]
                fig = plt.figure(figsize=(fig_width, 6))
                gs = gridspec.GridSpec(dims[0], dims[1])
                ax = fig.add_subplot(gs[1, :])
                graphic_record = GraphicRecord.from_biopython_record(record, fun_label=fun_label, fun_color=fun_color, features_filter=features_filter)
                try:
                    graphic_record.plot(fig_width=fig_width, ax=ax)
                except np.linalg.linalg.LinAlgError:
                    logger.error("Could not diagram: %s" % record.id)

                cluster_silhouettes = self.cluster_silhouette_map[id]
                cluster_lineages = self.cluster_lineage_map[id]
                ax.text(0, 0.85, "c", transform=ax.transAxes, weight='bold', fontname='Helvetica', fontsize=12)

                # SILHOUETTE
                ax = fig.add_subplot(gs[0, 0])
                self.plot_cluster_silhouettes(cluster_silhouettes, ax=ax)
                ax.text(-0.2, 1.1, "a", transform=ax.transAxes, weight='bold', fontname='Helvetica', fontsize=12)

                # TAXONOMY BAR CHART
                ax = fig.add_subplot(gs[0, 2])
                self.plot_cluster_composition(cluster_lineages, ax=ax)
                ax.text(-0.2, 1.1, "b", transform=ax.transAxes, weight='bold', fontname='Helvetica', fontsize=12)

                # TEXT
                plt.text(5.5, 0, self.get_contig_description_text(id, cluster_lineages=cluster_lineages), fontsize=6, fontname='Helvetica')

                # SAVE
                file_name = self.get_diagram_filename(id)
                fig.savefig(file_name)
                plt.close(fig)

                self.have_been_diagramed += [id]

    def make_contig_plots(self):
        """
        This function makes plots of contigs witout the diagrams
        :return: None
        """
        output_dir = self.get_diagram_output_directory()
        if output_dir and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if self.ids_to_diagram is None:
            self.ids_to_diagram = []
        if self.have_been_diagramed is None:
            self.have_been_diagramed = []

        for id in set(self.ids_to_diagram) - set(self.have_been_diagramed):
            logger.info("Plotting: %s..." % id)
            contig_features = self.contig_features[self.contig_ids == id]
            appended_data = np.vstack((self.phage_features, contig_features))
            assignments = learning.kmeans(appended_data, self.k_clusters, verbose=False)
            cluster = assignments[-1]
            cluster_phage = np.arange(self.num_reference_phage)[assignments[:-1] == cluster]
            cluster_lineages = np.array([self.lineages[i] for i in cluster_phage])
            cluster_silhouettes = learning.cluster_silhouettes(appended_data, assignments, assignments[-1])

            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 5)

            # SILHOUETTE
            ax = fig.add_subplot(gs[0, 0])
            self.plot_cluster_silhouettes(cluster_silhouettes, ax=ax)

            # TAXONOMY BAR CHART
            ax = fig.add_subplot(gs[0, 2])
            self.plot_cluster_composition(cluster_lineages, ax=ax)

            # TEXT
            plt.text(5.5, 0, self.get_contig_description_text(id, cluster_lineages=cluster_lineages), fontsize=6)

            # SAVE
            file_name = self.get_diagram_filename(id)
            fig.savefig(file_name)
            plt.close(fig)

            self.have_been_diagramed += [id]

    def get_contig_description_text(self, id, cluster_lineages=None, lineage_depth=6, line_limit=15):
        """
        This function makes a text description for a given contig id
        :param id: A contig ID to get information for
        :param cluster_lineages: A list of lineages
        :return: A string that contains all information about the contig
        """
        text = self.id_header_map[id].strip()
        if self.dataset_name:
            text += "\nDataset: %s" % self.dataset_name

        if id in self.contig_category_map.keys():
            category = self.contig_category_map[id]
            category_name = get_category_name(category)
        else:
            category = "No VirSorter Category"
            category_name = ""
        text += "\nCategory: {cat_num} - {cat_name}".format(cat_num=category, cat_name=category_name)

        if id in self.phamer_dict.keys():
            phamer_score = self.phamer_dict[id]
        else:
            phamer_score = "Not scored"
        text += "\nPhamer score: {score}".format(score=phamer_score)

        if id in self.taxonomy_prediction_dict:
            text += "\nCluster enriched: %s" % self.taxonomy_prediction_dict[id][1]

        # IMG genes part
        lines = []
        if self.img_summary is not None and os.path.exists(self.img_summary):
            f = open(self.img_summary, 'r')
            lines = [line for line in f.readlines() if line.startswith(str(id) + ',')]
            f.close()
        if len(lines) > 0:
            text += "\nIMG Annotations:"
        i = 1
        phylogenies = []
        for line in lines:
            product = ','.join(line.split(', ')[2:-1])
            phylogeny = line.split(', ')[-1]
            phylogenies.append(phylogeny.split(';'))
            if text.count('\n') > line_limit:
                text += "\n More genes listed in: %s" % os.path.basename(self.img_summary)
                break
            elif not 'missing' in phylogeny.lower() and not 'missing' in product.lower():
                text += "\n %d. %s" % (i, product)
                if 'virus' in phylogeny.lower() or 'phage' in phylogeny.lower():
                    text += " (Viral)"
            i += 1

        # lineage proportion part
        phylogenies = tax.extend_lineages(phylogenies)
        text += "\nGene Homology Proportions:\n"
        lineage_proportion_dict_list = tax.lineage_proportions(phylogenies)
        maximum_depth = min(tax.deepest_classification(phylogenies), 5)
        for depth in xrange(maximum_depth):
            prop_dict = lineage_proportion_dict_list[depth]
            phylogeny = [p for p in prop_dict.keys() if prop_dict[p] == max(prop_dict.values())][0]
            percentage = prop_dict[phylogeny]
            text += "%.1f%% %s%s" % (100 * percentage, phylogeny.strip(), [', ', ''][depth == maximum_depth])

        text = text.strip()
        if text.endswith(","):
            text = text[:-1]
        text = text.strip()

        return text

    # Summary files
    def make_prediction_summary(self, args=None):
        """
        This function makes a summary for all the phage which were predicted by VirSorter and Phamer together
        :return: None. It saves this summary to file
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

        summary = basic.generate_summary(args, header="Summary of PhaMers compared to VirSorter", line_start='# ')
        # total_count = np.sum(category_counts[[1, 2, 4, 5]]) + np.sum(phamer_category_counts[[3, 6]])
        # summary += "\nTotal phages found by PhaMers AND VirSorter: %s\n" % total_count

        for category in xrange(1, 7):
            summary += "VirSorter Category %s: " % category
            summary += "%d (PhaMers: %d)\n" % (category_counts[category], phamer_category_counts[category])

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

    def make_integrated_summary(self):
        """
        This function makes a CSV summary of all the results from VirSroter, Phamer, and IMG
        :return: None
        """
        criteria = lambda id: self.virsroter_map[id] in [1, 2, 4, 5] or (id in self.phamer_dict.keys() and self.phamer_dict[id] >= self.phamer_score_threshold)
        ids = [id for id in self.virsroter_map.keys() if criteria(id)]
        df = pd.DataFrame(columns=self.summary_fields)
        df['Header'] = [self.id_header_map[id] for id in ids]
        df['Contig_ID'] = ids
        df['Contig_Name'] = [id_parser.get_contig_name(self.id_header_map[id]) for id in ids]
        df['Length'] = [id_parser.get_contig_length(self.id_header_map[id]) for id in ids]
        df['VirSorter_Category'] = [self.virsroter_map[id] for id in ids]
        if self.dataset_name is not None:
            df['DataSet'] = self.dataset_name

        if self.taxonomy_prediction_dict is None:
            logger.info("Clustering contigs with Phages to predict taxonomy...")
            self.taxonomy_prediction_dict = self.get_taxonomy_prediction_dict()

        for id in ids:
            try:
                df.Phamer_Score[df.Contig_ID == id] = self.phamer_dict[id]
            except KeyError:
                pass

            try:
                df.IMG_Products_and_Phylogenies[df.Contig_ID == id] = self.img_proucts_map[id]
            except KeyError:
                pass

            if id in self.taxonomy_prediction_dict:
                df.PhaMers_Taxonomy[df.Contig_ID == id] = self.taxonomy_prediction_dict[id][1]

        summary_file_path = self.get_prediction_summary_filename()
        df.to_csv(summary_file_path, delimiter=', ')

        self.output_fasta_file = self.get_output_fasta_filename()
        logger.info("Saving putative phages in FASTA format: %s" % os.path.basename(self.output_fasta_file))
        fileIO.extract_fasta_by_id(self.fasta_file, ids, self.output_fasta_file)

    def make_gene_csv(self):
        """
        This function parses files from a directory that contains a IMG COG, phylogeny, and product files and writes the
        compiled data of contig id, phylogenies, and products into a tab-separated summary file
        :return: None
        """
        if self.img_directory is None or not os.path.isdir(self.img_directory):
            logger.warning("No IMG directory, cannot parse IMG files.")
            return
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
        if self.fasta_file is not None:
            self.id_header_map = get_id_header_map(self.fasta_file)
        else:
            logger.warning("No fasta file...")
            self.id_header_map = None

        logger.debug("Loading features (%s)..." % os.path.basename(self.features_file))
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

        logger.debug("Generating truth table...")
        self.truth_table = self.get_virsorter_phamer_truth_table(strict=self.strict)
        self.virsroter_map = {phage.id: phage.category for phage in fileIO.read_virsorter_file(self.virsorter_summary)}

        logger.debug("Loading contig names and ids...")
        self.contig_name_id_nap = get_name_id_map(contigs_file=self.fasta_file)
        logger.debug("Loading IMG data...")
        if self.img_directory is not None and os.path.isdir(self.img_directory):
            self.img_proucts_map = img.get_contig_map(self.img_directory, self.contig_name_id_nap)
        else:
            logger.info("No IMG directory found.")
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

    def get_virsorter_phamer_truth_table(self, categories=None, strict=True):
        """
        This function finds the true positives, false positives, false negatives, and true positives for phage
        predictions from PhaMers and from VirSorter
        :param categories: Give a list of VirSorter categories for which phage should only be counted as putative
        predictions if they fit these categories
        :return: A tuple of lists of IDs for true positives, false positives, false negatives, and true positives in that order
        """
        phamer_ids = [id for id in self.phamer_dict.keys() if self.phamer_dict[id] >= self.phamer_score_threshold]

        vs_phage = fileIO.read_virsorter_file(self.virsorter_summary)
        for phage in vs_phage:
            if phage.id in self.phamer_dict.keys():
                phage.phamer_score = self.phamer_dict[phage.id]

        if categories is None:
            if strict:
                categories = [1, 2, 4, 5]
            else:
                categories = [1, 2, 3, 4, 5, 6]

        vs_ids = [phage.id for phage in vs_phage if categories is None or phage.category in categories]

        true_positives = [id for id in self.phamer_dict.keys() if id in vs_ids and id in phamer_ids]
        false_positives = [id for id in self.phamer_dict.keys() if id in phamer_ids and id not in vs_ids]
        false_negatives = [id for id in self.phamer_dict.keys() if id in vs_ids and id not in phamer_ids]
        true_negatives = [id for id in self.phamer_dict.keys() if id not in phamer_ids and id not in vs_ids]

        return true_positives, false_positives, false_negatives, true_negatives

    def get_taxonomy_prediction_dict(self):
        """
        This function makes a map from id to taxonomic prediction based on clusters
        :return: A dictionary mapping ids to taxonomic prediction
        """
        self.taxonomy_prediction_dict = {}
        self.cluster_silhouette_map = {}
        self.cluster_lineage_map = {}
        ids = self.get_virsorter_ids()
        j = 1
        for id in ids:
            if self.ids_to_diagram is not None and id not in self.ids_to_diagram:
                    logger.info("Skipping id: %s" % id)
                    continue
            logger.debug("ID: %s (%d of %d contigs)" % (id, j, len(ids)))
            j += 1
            contig_features = self.contig_features[self.contig_ids == id]
            appended_data = np.vstack((self.phage_features, contig_features))
            assignments = learning.kmeans(appended_data, self.k_clusters, verbose=False)
            cluster = assignments[-1]
            cluster_phage = np.arange(self.num_reference_phage)[assignments[:-1] == cluster]
            self.cluster_lineage_map[id] = np.array([self.lineages[i] for i in cluster_phage])
            self.cluster_silhouette_map[id] = learning.cluster_silhouettes(appended_data, assignments, assignments[-1])

            cluster_size = len(cluster_phage)
            if cluster_size > 0:
                for lineage_depth in xrange(5, -1, -1):
                    tup = tax.find_enriched_classification(self.cluster_lineage_map[id], self.lineages, lineage_depth)
                    if None not in tup:
                        sil = self.cluster_silhouette_map[id][-1]
                        mean_sil = np.mean(self.cluster_silhouette_map[id][:-1])
                        std_sil = np.std(self.cluster_silhouette_map[id][:-1])
                        if sil < max(0, mean_sil - std_sil):
                            continue
                        tax_text = "{pct} {kind} ({taxon}), sil:{sil} ({mean_sil} +/- {std_sil}), p={p}"
                        tax_text = tax_text.format(kind=tup[0], pct=100.0*tup[2], sil=sil, mean_sil=mean_sil, std_sil=std_sil, p=tup[1][1], taxon=self.phylogeny_names[min(4, lineage_depth)])
                        self.taxonomy_prediction_dict[id] = (tup, tax_text)
                        break
        return self.taxonomy_prediction_dict

    def get_virsorter_ids(self):
        """
        This function finds VirSorter IDs for all putative phages
        :return: A list of IDs
        """
        f = open(self.virsorter_summary)
        ids = [id_parser.get_id(line.split(',')[0]) for line in f.readlines() if not line.startswith('#')]
        f.close()
        return ids

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
            else:
                for potential_file in fasta_files:
                    if not os.path.splitext(potential_file)[0].endswith("genes"):
                        self.fasta_file = os.path.join(self.input_directory, potential_file)
                        break
                if self.fasta_file is None:
                    self.fasta_file = os.path.join(self.input_directory, fasta_files[0])

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
            else:
                logger.error("Count not find features file in %d" % self.input_directory)

            virsorter_directories = basic.search_for_file(self.input_directory, start='VirSorter')
            if len(virsorter_directories) == 1:
                self.virsorter_directory = virsorter_directories[0]
                self.find_virsorter_files()

            img_directories = basic.search_for_file(self.input_directory, start='IMG_')
            if len(img_directories) == 1:
                self.img_directory = img_directories[0]

        else:
            logger.error("Could not find directory: %s" % self.input_directory)
            exit()

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
        return os.path.join(self.diagram_output_directory, "contig_diagram_{id}.pdf".format(id=id))

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
        return os.path.join(self.pie_charts_output, 'close_phage_contig_{id}.pdf'.format(id=id))

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
        return os.path.join(self.output_directory, "category_boxplot.pdf")

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
        if self.output_directory is not None and self.img_directory is not None:
            return os.path.join(self.output_directory, '%s.summary' % os.path.basename(os.path.relpath(self.img_directory)))
        elif self.output_directory is not None:
            return os.path.join(self.output_directory, "IMG.sumary")
        else:
            return "IMG.summary"

    def get_roc_filename(self):
        """
        This function makes a filename for the roc curve comparing performance
        :return: A path to file to save an ROC curve
        """
        return os.path.join(self.output_directory, 'roc.pdf')

    def get_output_fasta_filename(self):
        return os.path.join(self.output_directory, "putative_phage.fasta")

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


def get_category_name(category):
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


def seq_features_to_color(seq_feature):
    """
    This function converts a seq feature into a particular color
    :param seq_feature: A BioPython SeqFeature object
    :return: A color that should be plotted for this feature
    """
    product = seq_feature.qualifiers['product'][0]
    hallmark_genes = ['terminase', 'capsid', 'portal', 'spike', 'tail', 'sheath', 'tube', 'mu'
                      'virion formation', 'coat', 'baseplate', 'integrase', 'phage']
    if any([gene_name in product.lower() for gene_name in hallmark_genes]):
        return [0, 1, 0]
    elif 'hypothetical' in product and 'protein' in product:
        return [1, 0, 0]
    elif 'cluster' in product:
        return [1, 1, 0]
    else:
        return [0, 0, 1]


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
    options_group.add_argument('-diagram', '--diagram_ids', nargs='+', help="Contig IDS to diagram.")
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
    logger.info("Loading data...")
    analyzer.load_data()

    analyzer.ids_to_diagram = args.diagram_ids

    if not os.path.isdir(analyzer.output_directory):
        os.mkdir(analyzer.output_directory)

    logger.info("Preparing GenBank files for SnapGene...")
    analyzer.prepare_gb_files_for_SnapGene()
    logger.info("Making prediction metrics file...")
    analyzer.make_prediction_summary(args=args)
    logger.info("Making taxonomic predictions...")
    analyzer.get_taxonomy_prediction_dict()
    logger.info("Making integrated summary...")
    analyzer.make_integrated_summary()
    logger.info("Making gene csv file...")
    analyzer.make_gene_csv()
    logger.info("Making plots...")
    analyzer.make_all_plots()

    logger.info("Analysis complete.")