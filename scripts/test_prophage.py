#!/usr/bin/env python


import sys, kmer, phamer

if __name__ == '__main__':

    phage_kmer_file = '/Users/jonpdeaton/Dropbox/Documents/research/phamer/data/phage_4mer_counts.csv'
    contigs_file = '/Users/jonpdeaton/Dropbox/Documents/research/phamer/inputs/super_contigs.BijahRoadSide4.fasta'

    header, contigs = kmer.read_fasta(contigs_file)
    phage_kmers = kmer.read_kmer_file(phage_kmer_file)
    phage_kmers = kmer.normalize_counts(phage_kmers)
    long_contigs = [contig for contig in contigs if len(contig) >= 7500]

    assignment = phamer.dbscan(phage_kmers, 0.012, 2, verbose=True)

    for contig in long_contigs:
        start, stop = phamer.most_likely_prophage(contig, phage_kmers, assignment=assignment)
        print "start, stop = %d, %d (%.1f kbp contig)" % (start, stop, (len(contig)) / 1000.0)
