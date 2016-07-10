#!/usr/bin/env python

import os
import argparse
import gzip
from Bio import SeqIO

def make_database(reference_fasta, databse_name):
    command = 'makeblastdb -in %s -dbtype nucl -out %s' % (reference_fasta, databse_name)
    os.system(command)

def blast_fasta(fasta, database, output):
    temporary = False
    if not output:
        output = "/tmp/temp_blast_output.%d.txt" % os.getpid()
        temporary = True

    blast_command = 'blastn -query %s -db %s -out %s' % (fasta, database, output)
    blast_command += ' -evalue 1 -gapopen 3 -gapextend 1 -window_size 100'
    os.system(blast_command)

    handle = open(output, 'r')
    contents = handle.read()
    handle.close()

    if temporary:
        os.remove(output)

    return contents

def make_query(contigs_file, id, output):
    if contigs_file.endswith('.gz'):
        handle = gzip.open(contigs_file, 'r')
    else:
        handle = open(contigs_file, 'r')
    records = SeqIO.parse(handle, 'fasta')
    i = 0
    for record in records:
        if get_contig_id(record.id) == id:
            output_handle = open(output, 'w')
            SeqIO.write(record, output_handle, 'fasta')

def get_contig_id(header):
    '''
    Returns the ID number from a contig header
    :param header: The contig header
    :return: The contig ID number as an integer
    '''
    parts = header.split('_')
    return int(parts[1 + parts.index('ID')].replace('-circular', ''))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('contigs_file', type=str, help='Fasta file containing contigs')
    parser.add_argument('-id', type=int, help='Contig id to blast')
    parser.add_argument('-db', '--database', type=str, help='Blast database')
    parser.add_argument('-ref', '--reference', type=str, help='Reference fasta to make a database from')
    parser.add_argument('-out', '--output', type=str, help='Output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    contigs_file = args.contigs_file
    id = args.id
    database = args.database
    reference = args.reference
    output = args.output
    verbose = args.verbose

    if not id:
        exit('Provide a contig ID with the -id argument')

    if not reference and not database:
        exit('Provide a reference fasta with -ref or BLAST database with -db')

    if reference:
        database = "%s.db" % reference
        make_database(reference, database)

    query_file = "%s.blast_query_ID_%d.fasta" % (os.path.basename(contigs_file), id)
    make_query(contigs_file, id, query_file)

    blast_results = blast_fasta(query_file, database, output)
    os.remove(query_file)
    if verbose:
        print blast_results