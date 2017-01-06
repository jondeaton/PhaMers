# PhaMers

This repository contains code for running PhaMers, a phage identification tool. This main scripts in this repository are the following:


## Dependencies

1. Python 2.7
2. Anaconda2
	a. Numpy
	b. Pandas
	c. MatPlotLib
	d. SciPy
	e. BioPython
3. dna_features_viewer

# Tools


1) phamer.py - This script takes a fasta file as an input, and scores each sequence by it's comparing k-mer frequencies to those of reference phage and bacterial genomes. Positive scores indicate more phage-like whereas negative scores indicate a sequence is less phage-like. This is done using basic machine learning methods on k-mer frequency vectors. This script also performs t-SNE on the k-mer frequency vectors so that contigs can be visually inspected and compared to reference genomes.

2) analysis.py - This script takes scores generated from phamer.py, phage prediction data from VirSroter, and gene annotation files from the Integrated Microbial Genomes (IMG) annotation pipeline and outputs a bunch of files summaryzing the annotation results, phamer prediction results, and genbank files that can be used to visualize contig genetic features.

3) feature_taxonomy.py - This script examines the relationship between k-mer frequencies and phage taxonomic classification

4) cross_validate.py - This script uses N-fold cross validaiton to quantify how well phage DNA sequences can be distinguished from non-phage genomes using the scoring method applied in phamer.py.



### Example Graphs


#### Contig Diagram

![contig_diagram_69](https://cloud.githubusercontent.com/assets/15920014/21732965/014ae34e-d411-11e6-9daf-685a9fa81ce5.png)


#### t-SNE Plot

![tsne_comparison](https://cloud.githubusercontent.com/assets/15920014/21732968/04422922-d411-11e6-8a92-b7636b412361.png)



