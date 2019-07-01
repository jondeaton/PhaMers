# PhaMers (*Pha*ge k-*mers*)

This repository contains the implementation of "PhaMers"- a bioinformatic
tool for identifying bacteriophages (phages) from metagenomic sequenging data
on the basis of their k-mer frequencies. PhaMers uses basic techniques from
supervised machine learning with k-mer frequency vectors as a the feature
representaton. The PhaMers classificaiton algorithm is trained on k-mer
feature vectors from GenBank and RefSeq.

This repository also contains utilities to analyze and plot DNA sequences
to facilitate understanding of metagenomic datasets.

## Installation
PhaMers requires Python 2.7 as well as some basic packages which
may be installed by running the following command

    python setup.py install

## Usage
To score DNA sequences using PhaMers, use the following command

    python scripts/phamer.py -in $input_dir -data data --debug --equalize_reference

where the variable `$input_dir` is a path to a directory that contains a collection
of FASTA files with sequences that will be scored. This will create a directory
called `phamer_output` containing the scores in a file called `phamer_scores.csv`.
Scores range from -1 to 1 with higher scores indicating that the sequence
is more phage-like.

To run further analysis and generate plots, run the following command

    python scripts/analysis.py -in $input_dir -data data --debug

with the variable `$input_dir` as before.

## Scripts
This repository contains quite a few different scripts, which
are now briefly described:

- phamer.py
    - The main PhaMers scoring funcitonality is contained here. 
    This script can take in files in fasta format, count k-mers, and 
    score files against referece datasets. This script can also do t-SNE
    on the combined datasets.

- analysis.py
    - This script integrats and presents data from PhaMers, VirSorter, and
     IMG. This script makes t-SNE plots of metagenomics datasets, contig diagrams, 
     performance plots, and text files that summarize results.

- feature_taxonomy.py
    - A class and functions that do t-SNE and cluster points to examine 
    enrichmet for taxa.

- cross_validate.py
    - A class and functions that help to do N-Fold cross validation

- kmer.py
    - Functions for counting k-mers.

- cluster.py
    - Cluster optimization analysis.

- learning.py
    - Some functions that implement several tools useful for machine learning and some wrapper functions for Scipy ML functions.

- distinguishable_colors.py
    - Some functions for getting a set of colors that are able to be 
    distinguished from eachoter visually.

- fileIO.py
    - Some functions for getting data in and out of files that score inputs and outputs for PhaMers

- id_parser.py
    - Functions that help parse headers of different formats to turn them in to IDs.

- img_parser.py

    - Functions for parsing IMG output files.

- basic.py
    - Some basic utility functions that might be useful in any program