# PhaMers

This repository contains code for running PhaMers, a phage identification tool. PhaMers uses k-mer frequencies and basic machine learning algorithms to predict whether an unidentified DNA sequence is likely that of a phage of not. PhaMers also graphs DNA sequences and makes plots that facilitate understanding of metagenomic datasets.


## Dependencies

1. Python 2.7
2. Anaconda2
	a. Numpy
	b. Pandas
	c. MatPlotLib
	d. SciPy
	e. BioPython
3. dna_features_viewer

## Tools

--> phamer.py - The main PhaMers scoring funcitonality is contained here. This script can take in files in fasta format, count k-mers, and score files against referece datasets. This script can also do t-SNE on the combined datasets.

--> analysis.py - This script integrats and presents data from PhaMers, VirSorter, and IMG. This script makes t-SNE plots of metagenomics datasets, contig diagrams, performance plots, and text files that summarize results.

--> feature_taxonomy.py - A class and functions that do t-SNE and cluster points to examine enrichmet for taxa.

--> cross_validate.py - A class and functions that help to do N-Fold cross validation

--> kmer.py - Functions for counting k-mers.

--> cluster.py - Cluster optimization analysis.

--> learning.py - Some functions that implement several tools useful for machine learning and some wrapper functions for Scipy ML functions.

--> distinguishable_colors.py - Some functions for gettting a set of colors that are able to be distinguished from eachoter visually.

--> fileIO.py - Some functions for getting data in and out of files that score inputs and outputs for PhaMers

--> id_parser.py - Functions that help parse headers of different formats to turn them in to IDs.

--> img_parser.py - Functions for parsing IMG output files.

--> basic.py - Some basic utility functions that might be useful in any program


### Example Graphs

#### Contig Diagram

![contig_diagram_69](https://cloud.githubusercontent.com/assets/15920014/21732965/014ae34e-d411-11e6-9daf-685a9fa81ce5.png)


#### t-SNE Plot

![tsne_comparison](https://cloud.githubusercontent.com/assets/15920014/21732968/04422922-d411-11e6-8a92-b7636b412361.png)





