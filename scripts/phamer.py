#!/usr/bin/env python
'''
Phamer: Phage-finding algorithm that uses k-mer frequency comparison and t-SNE
Jonathan Deaton, Quake Lab, Stanford University, 2016
'''

import os
import argparse
import time
import warnings
import numpy as np
import tsne
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy import stats
from sklearn.neighbors.kde import KernelDensity
from sklearn import svm
import matplotlib
matplotlib.use('Agg')
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
import logging

# My stuff
import kmer

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def kmeans(data, k):
    '''
    K-Means clustering wrapper function
    :param data: The data to cluster as a numpy array with datapoints being rows
    :param k: The number of clusters
    :return: A numpy array with elements corresponsing to the cluster assignment of each point
    '''
    assignment = KMeans(n_clusters=k).fit(data).labels_
    ss = silhouette_score(data, assignment)
    logger.info("K-means clustering (k=%d) silhouette score: %f" % (k, np.mean(ss)))
    return assignment


def silhouettes(data, assignment):
    '''
    This function returns the silhouette values for datapoints given a particular cluster assignment, which are values
    from -1 to 1 that describe how well the datapoints are assigned to clusters
    :param data: A numpy array of row vectors specifying the datapoints
    :param assignment: A list of integers specifying the cluster that each datapoint was assigned to. -1 = not assigned
    :return: A numpy array of silhouette values corresponding to each datapoint
    '''
    return silhouette_samples(data, assignment)


def cluster_silhouettes(data, assignment, cluster):
    '''
    This function computes the silhouette values for a single cluster
    :param data: A numpy array of row vectors specifying the datapoints
    :param assignment: A list of integers specifying the cluster that each datapoint was assigned to. -1 = not assigne
    :param cluster: The cluster to calculate silhouette values for
    :return: A numpy array with silhouette values for each point within the cluster
    '''
    ss = silhouettes(data, assignment)
    return np.array([ss[i] for i in xrange(len(assignment)) if assignment[i] == cluster])


def dbscan(data, eps, min_samples, expected_noise=None):
    '''
    DBSCAN wrapper function
    :param data: A numpy with rows that are datapoints in a vector space
    :param eps: A float specifying the maximum distance that a point can be away from a cluster to be included
    :param min_samples: The minimum number of samples per cluster
    :param expected_noise: An optional parameter specifying the expected amount of noise in the clustering. Passing a
            value in this argument will cause eps to change until noise is within 5% of the specified value
    :return: An array specifying the cluster assignment of each data-point
    '''
    if expected_noise:
        asmt = dbscan(data, eps, min_samples)
        noise = float(np.count_nonzero(asmt == -1)) / data.shape[0]
        error = noise - expected_noise
        if abs(error) >= 0.05:
            eps *= 1 + (error * (0.5 + (0.2 * np.random.rand())))
            asmt = dbscan(data, eps, min_samples, expected_noise=expected_noise)
    else:
        asmt = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
        num_unassigned = np.count_nonzero(asmt == -1)
        tup = (eps, min_samples,  1+max(asmt), num_unassigned, len(asmt), 100 * float(num_unassigned) / data.shape[0])
        logger.info("Clustered, eps:%f, mpts:%d, %d clusters, %d/%d (%.1f%%) noise" % tup)
    return asmt


def knn(query, data, labels, k=3):
    '''
    K-Nearest Neighbors wrapper method
    :param query: The point to search a label for as a numpy array
    :param data: The data to compare the query to as a numpy array where rows are points
    :param labels: The labels of each point in the data array
    :param k: Number of nearest neighbors to consider
    :return: The guessed classification
    '''
    near_labels = labels[np.argsort(distances(query, data))[:k]]
    m = int(stats.mode(near_labels)[0])
    r = float(np.count_nonzero(near_labels == m)) / k
    return m


def get_density(point, data, bandwidth=0.1):
    '''
    This function returns the density of the data at the given point, using t-distribution kernel density
    :param point: A numpy array vector specifying a point in space to evaluate the density at
    :param data: A 2D numpy array of points (rows)
    :return: A float representing the density of datapoints at the given point
    '''
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    return kde.score_samples(np.array([point]))[0]


def get_centroids(data, assignment, start=0):
    '''
    A function for getting the centroids of clustered data-points
    :param data: The data-points as rows in a 2 dimensional numpy array
    :param assignment: A numpy array containing the cluster assignments  0 - num_centroids
    :return: A numpy array with rows being the centroid points for each cluster of data-points
    '''
    num_centroids = max(assignment)
    if num_centroids == -1:
        exit('Could not get centroids as data was not clustered. Exiting.')
    centroids = np.zeros((num_centroids, data.shape[1]))
    for i in xrange(start, num_centroids):
        centroids[i, :] = np.mean(data[assignment == i][:], axis=0)
    return centroids


def cluster_deviations(data, assignment):
    '''
    A function for getting mean average distance (MAD) from the centroid of a number of clustered data-points
    :param data: The data-points in a 2 dimensional numpy array
    :param assignment: A clustering assignment array for each of the datapoints
    :return: A numpy array containing the MAD for the points from each cluster
    '''
    num_clusters = max(assignment)
    centroids = get_centroids(data, assignment)
    deviations = np.zeros(num_clusters)
    for cluster in xrange(num_clusters):
        which = [i for i in xrange(data.shape[0]) if assignment[i] == cluster]
        deviations[cluster] = np.mean(distances(centroids[cluster], data[which]))
    return deviations


def distances(vector, data):
    '''
    A function for finding the distances from one point to many
    :param vector: A numpy array specifying the point in space to find distances from.
    :param data: A list of points in a numpy array to find distances to
    :return: A numpy array with each element being the distance from the vector to the datapoints
    '''
    if len(vector.shape) == 1:
        vector = np.array([vector])
    if vector.shape[0] != 1:
        exit('Error: Too many vectors passed to phamer.distances. pass a single row vector only')
    return np.linalg.norm(np.repeat(vector, data.shape[0], axis=0) - data, axis=1)


def closest_to(point, picks):
    '''
    This function returns one point from many which is closest to another point
    :param point: A numpy array specifying the point of interest to compare proximity of all other points
    :param picks: A numpy array with rows specifying other points to pick the closest from
    :return: A numpy array which is the row of the "picks" array that is closest to the "point" vector
    '''
    return picks[np.argmin(distances(point, picks))]


def chop(array, chops):
    '''
    This function is for separating the rows of a numpy array
    i.e. chop(X, [10, 15]) will return a list in which the first element is a numpy array containing the first 10 rows of X,
    and the second element is a numpy array containing next 15 rows of X.
    :param array: The array to chop up
    :param chops: A list of integers to chop the array into
    :return: A list of numpy arrays split as described previously
    '''
    chopped = []
    array = np.array(array)
    at = 0
    for n in chops:
        chopped.append(array[at:at+n])
        at += 1 + n
    return chopped


def most_likely_prophage(sequence, phage_kmers, assignment=None):
    '''
    This function finds the 7.5kbp region of a sequence most likely to be a prophage within a sequence
    :param sequence: A string representing the DNA sequence of interest
    :param phage_kmers: A numpy array containing the k-mer counts of
    :param assignment: An optional parameter specifying a clustering assignment of the phage k-mer data-points
    :return: A tuple containing the start and stop indicies of the most likely prophage region in the sequence
    '''
    window = 7500
    slide = 100
    if len(sequence) <= window:
        return (0, len(sequence))

    if assignment is None:
        assignment = dbscan(phage_kmers, 0.017571, 16, expected_noise=0.35)

    phage_centroids = get_centroids(phage_kmers, assignment)
    kmer_length = int(np.log(phage_kmers.shape[1]) / np.log(4))

    num_windows = 1 + (len(sequence) - window) / slide
    slides_per_window = window / slide
    num_slides = 1 + len(sequence) // slide
    slide_kmers = np.zeros((num_slides, phage_kmers.shape[1]))
    for i in xrange(num_slides):
        slide_kmers[i] = kmer.count(sequence[slide * i:slide * (i + 1)], kmer_length, normalize=False)

    scores = np.zeros(num_windows)
    for i in xrange(num_windows):
        window_kmer_freqs = kmer.normalize_counts(np.sum(slide_kmers[i:i+slides_per_window], axis=0))
        scores[i] = get_density(window_kmer_freqs, phage_centroids)
    del slide_kmers

    start = np.argmax(scores)
    stop = start + slides_per_window

    return (slide * start, slide * stop)


def score_points(points, positive_data, negative_data, method='combo', eps=[0.012,0.012], min_samples=[2,2]):
    '''
    This function scores a set of points against positive and negative datapoints.
    :param points: The points to score as row vectors in a numpy array
    :param positive_data: The positive datapoints to use in scoring as row vectors in a numpy array
    :param negative_data: The negative datapoints to use in scoring as row vectors in a numpy array
    :param eps: A list specifying the DBSCAN proximity length for positive and negative in that order
    :param min_samples: A list specifying the DBSCAN minimum points in a cluster for positive and negative in that order
    :return: A tuple containing a list of ...
    '''
    num_points = points.shape[0]
    num_positive = positive_data.shape[0]
    num_negative = negative_data.shape[0]
    train = np.vstack((positive_data, negative_data))
    labels = np.append(np.ones(num_positive), np.zeros(num_negative))

    logger.info("Scoring %d points with: %s..." % (points.shape[0], method.upper()))

    if method == 'dbscan':
        positive_assignment = dbscan(positive_data, eps[0], min_samples[0])
        negative_assignment = dbscan(negative_data, eps[1], min_samples[1])
        if max(positive_assignment) == -1:
            positive_assignment = kmeans(positive_data, 86)
        if max(negative_assignment) == -1:
            negative_assignment = kmeans(negative_data, 86)

        positive_centroids = get_centroids(positive_data, positive_assignment)
        negative_centroids = get_centroids(negative_data, negative_assignment)
        scores = [score_point(point, closest_to(point, positive_centroids), closest_to(point, negative_centroids)) for point in points]
    elif method == 'kmeans':
        positive_assignment = kmeans(positive_data, 30)
        negative_assignment = kmeans(negative_data, 30)
        positive_centroids = get_centroids(positive_data, positive_assignment)
        negative_centroids = get_centroids(negative_data, negative_assignment)
        scores = [score_point(point, closest_to(point, positive_centroids), closest_to(point, negative_centroids)) for point in points]
    elif method == 'svm':
        machine = svm.NuSVC()
        machine.fit(train, labels)
        scores = machine.predict(points)
    elif method == 'knn':
        scores = [knn(point, train, labels) for point in points]
    elif method == 'density':
        scores = np.zeros(num_points)
        for i in xrange(num_points):
            point = points[i]
            pos_density = get_density(point, positive_data, bandwidth=0.005)
            neg_density = get_density(point, negative_data, bandwidth=0.01)
            score = pos_density - neg_density
            scores[i] = score
    elif method == 'silhouette':
        positive_appended = np.append(positive_data, points, axis=0)
        negative_appended = np.append(negative_data, points, axis=0)
        positive_assignment = kmeans(positive_appended, 86)
        negative_assignment = kmeans(negative_appended, 86)
        pos_sils = silhouettes(positive_appended, positive_assignment)
        neg_sils = silhouettes(negative_appended, negative_assignment)
        scores = np.array(pos_sils[-num_points:] - neg_sils[-num_points:])
    elif method == 'combo':
        knn_scores = score_points(points, positive_data, negative_data, method='knn')
        cluster_scores = score_points(points, positive_data, negative_data, method='dbscan')
        scores = 2 * ((knn_scores - 0.5) + 150 * cluster_scores)
    return np.array(scores)


def score_point(point, nearest_positive, nearest_negative):
    '''
    This function scores a point for being near positive and away from negative
    :param point: A numpy array with the point to be scored
    :param nearest_positive: A numpy array containing the centeroid of the nearest positive cluster
    :param nearest_negative: A numpy array containing the centeroid of the nearest negative cluster
    :return: A score for that point. A higher score is if the point is near the positive and far from the negative
    '''
    return np.exp(np.linalg.norm(point - nearest_negative) - np.linalg.norm(point - nearest_positive)) - 1


def get_contig_id(header):
    '''
    Returns the ID number from a contig header
    :param header: The contig header
    :return: The contig ID number as an integer
    '''
    parts = header.split('_')
    return int(parts[1 + parts.index('ID')].replace('-circular', ''))


def get_bacteria_id(header):
    '''
    This function gets the ID from a bacteria fasta header
    :param header: The header string
    :return: The genbank id
    '''
    return header.split(' ')[0]


def get_phage_id(header):
    '''
    This function gets the genbank id from the header of phage fasta
    :param header: The string fasta header
    :return: The string representation of the id
    '''
    return header.split('|')[3]


def get_id(header):
    '''
    Decides what kind of ID needs to be retrieved and decides how to correctly parse it
    :param header: the header to be parsed
    :return: The ID which has been retrieved from the header
    '''
    if '_ID_' in header:
        return get_contig_id(header)
    elif header.count('|') == 4:
        return get_phage_id(header)
    else:
        return get_bacteria_id(header)


def save_tsne_data(filename, tsne_data, ids, header='tsne output\nid,x,y'):
    '''
    This function saves t-SNE data to file with ids in the first column and x,y values in the second and third
    :param filename: The name of the file to save the data to
    :param tsne_data: x,y coordinates for each poitn to store
    :param ids: The ids coresponding to each datapoint in the tsne_data
    :param header: The header of the file
    :return: None
    '''
    X = np.hstack((np.array([ids]).transpose(), tsne_data.astype(str)))
    np.savetxt(filename, X, fmt='%s', delimiter=',', header=header)


def read_tsne_file(tsne_file):
    '''
    This file returns the data from a t-SNE file, which can be split into the kinds that is
    :param tsne_file: The filename of the t-SNE csv file
    :param chop: Set to true to have the data split into a list based on the number of unknown, positive, and negative
    datapoints. Number of each kind are specified on the second line of the file as follows: #unk,pos,neg=(n_unk, n_pos, n_neg)
    :return: Either the raw data in a numpy array, or if chopped, a list of numpy arrays in the order unknown, positive
    and then negative
    '''
    data = np.loadtxt(tsne_file, dtype=str, delimiter=',')
    ids = list(data[:, 0].transpose())
    ids = [int(id) for id in ids if represents_int(id)]
    points = data[:, 1:].astype(float)
    with open(tsne_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                nums = line.split('=')[1].replace('(','').replace(')','').strip()
                chops = [int(x.strip()) for x in nums.split(',')]
    return ids, points, chops


def represents_int(s):
    '''
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


def make_plots(positive, negative, unknown, filename='tsne_plot_all.png'):
    '''
    This function makes a t-SNE plot of all points
    :param positive: The 2D t-SNE positive (phage) points to plot in a numpy array
    :param negative: The 2D t-SNE negative points to plot in a numpy array
    :param unknown: The 2D t-SNE unknown points to plot in a numpy array
    :param filename: The name of the file to save the image to
    :return: None
    '''
    plt.figure(figsize=(30, 24))
    pos = plt.scatter(positive[:, 0], positive[:, 1], c=[0, 0, 1], label='positive', alpha=0.9, marker='o')
    neg = plt.scatter(negative[:, 0], negative[:, 1], c=[0, 0, 0], label='negative', alpha=0.9, marker='o')
    unk = plt.scatter(unknown[:, 0], unknown[:, 1], c=[1, 0, 0], label='unknown', alpha=0.9,  marker='o')
    plt.legend(handles=[pos, neg, unk])
    plt.title('t-SNE output')
    plt.grid(True)
    plt.savefig(filename)


def generate_summary(args):
    '''
    This makes a summary for the output file
    :param args: The parsed argument parser from function call
    :return: a beautiful summary
    '''
    return args.__str__().replace('Namespace(', '# ').replace(')', '').replace(', ', '\n# ').replace('=', ':\t') + '\n'


def read_phamer_output(filename):
    '''
    This file reads a phamer summary file and returns the scores for the Phamer run
    :param filename: The file name of the Phamer summary file
    :return: A distionary mapping contig ID to Phamer score
    '''
    score_dict = {}
    lines = open(filename, 'r').readlines()
    for line in lines:
        if '#' not in line:
            try:
                id = int(line.split()[1])
            except:
                id = get_contig_id(line.split()[1])
            score_dict[id] = float(line.split()[0])
    return score_dict


if __name__ == '__main__':

    call_time = time.strftime("%Y-%m-%d %H:%M")

    script_description='This script scores contigs based on k-mer frequency similarity'
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input_file', type=str, help='Fasta compilation file of unknown sequences')
    input_group.add_argument('-p', '--positive_input', type=str, help='Fasta compilation file of positive sequences')
    input_group.add_argument('-n', '--negative_directory', type=str, help='Dir')
    input_group.add_argument('-i', '--file_identifier', type=str, default='.fna', help='File identifier for fasta files in negative directory')
    input_group.add_argument('-pk', '--positive_kmer_file', type=str, help='positive kmer file')
    input_group.add_argument('-nk', '--negative_kmer_file', type=str, help='negative kmer file')
    input_group.add_argument('-kmers', '--input_kmer_file', type=str, default='', help="Input k-mer file")
    input_group.add_argument('-lin', '--lineage_file', type=str, default='phage_lineages.txt', help='Lineage file for positive sequences')
    input_group.add_argument('-tsne', '--tsne_file', default='tsne_out.csv', type=str, help='Preprocessed t-SNE data file in csv format')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output_dir', type=str, default='phamer_out', help='Directory to dump output files')

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-k', '--kmer_length', type=int, default=4, help='Length of k-mers analyzed')
    options_group.add_argument('-l', '--length_requirement', type=int, default=5000, help='Input sequence length requirement for scoring')

    tsne_options_group = parser.add_argument_group("t-SNE Options")
    tsne_options_group.add_argument('-dt', '--do_tsne', action='store_true', help='Flag to perform t-SNE')
    tsne_options_group.add_argument('-px', '--perplexity', type=float, default=30, help='t-SNE Perplexity')
    tsne_options_group.add_argument('-plot', action='store_true', default=False, help='Flag makes t-SNE plots')

    learning_options_group = parser.add_argument_group("Learning Options")
    learning_options_group.add_argument('-m', '--method', type=str, default='combo', help='Learning algorithm name')
    learning_options_group.add_argument('-eps', '--eps', type=float, default=2.1, help='DBSCAN eps parameter')
    learning_options_group.add_argument('-mp', '--minPts', type=int, default=2, help='DBSCAN minimum points per cluster parameter')

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', default=False, help='Debug console')

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    kmer_length = args.kmer_length
    length_requirement = args.length_requirement
    positive_file = args.positive_input
    positive_kmer_file = args.positive_kmer_file
    negative_directory = args.negative_directory
    negative_kmer_file = args.negative_kmer_file
    tsne_file = os.path.join(output_dir, args.tsne_file)
    perplexity = args.perplexity
    lineage_file = args.lineage_file
    do_tsne = args.do_tsne
    method = args.method
    input_kmer_file = args.input_kmer_file

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')


    if None not in [positive_kmer_file, negative_kmer_file] and os.path.exists(positive_kmer_file) and os.path.exists(negative_kmer_file):
        positive_ids, positive_kmer_count = kmer.read_kmer_file(positive_kmer_file, normalize=True)
        negative_ids, negative_kmer_count = kmer.read_kmer_file(negative_kmer_file, normalize=True)

        num_positive = positive_kmer_count.shape[0]
        num_negative = negative_kmer_count.shape[0]

        negative_kmer_count = negative_kmer_count[:num_positive]
        negative_ids = negative_ids[:num_positive]
        num_negative = num_positive
    else:
        positive_ids, positive_kmer_count = kmer.count


    if os.path.exists(input_kmer_file):
        logger.info("Reading k-mer counts from file: %s ..." % os.path.basename(input_kmer_file))
        unknown_ids, unknown_kmer_count = kmer.read_kmer_file(input_kmer_file)
        num_unknown = len(unknown_ids)
        which_are_long = [i for i in xrange(num_unknown) if np.sum(unknown_kmer_count[i]) >= length_requirement]
        unknown_ids = [unknown_ids[i] for i in which_are_long]

    else:
        # Screening input by length
        logger.info("Screening input by length %.1f kbp..." % (length_requirement / 1000.0))
        unknown_ids, unknown_sequences = kmer.read_fasta(input_file)
        num_unknown = len(unknown_ids)
        which_are_long = [i for i in xrange(num_unknown) if len(unknown_sequences[i]) >= length_requirement]
        unknown_ids = [unknown_ids[i] for i in which_are_long]
        logger.info("Done screening by input length.")

        # Counting Input
        logger.info("Counting k-mers...")
        contig_ids, unknown_kmer_count = kmer.count_file(input_file, kmer_length, normalize=False)
        if not os.path.isdir(output_dir):
            os.system('mkdir %s' % output_dir)

        kmer_out_file = os.path.join(output_dir, '%s_%dmers.csv' % (input_file, kmer_length))
        kmer_header = '%d-mers from %s' % (kmer_length, input_file)
        kmer.save_counts(unknown_kmer_count, contig_ids, kmer_out_file, args, header=kmer_header)


    unknown_kmer_count = kmer.normalize_counts(unknown_kmer_count)
    logger.info("done counting k-mers")

    unknown_kmer_count = unknown_kmer_count[which_are_long]

    num_positive = positive_kmer_count.shape[0]
    num_negative = negative_kmer_count.shape[0]
    num_unknown = unknown_kmer_count.shape[0]

    # Scoring Contigs
    scores = score_points(unknown_kmer_count, positive_kmer_count, negative_kmer_count, method=method)
    logger.info("done scoring.")

    # Summary file generation
    summary_header = '#Phamer summary and score file for %s from %s\n' % (input_file, call_time)
    f = open(os.path.join(output_dir, 'summary.txt'), 'w')
    f.write(summary_header)
    f.write(generate_summary(args))
    for score, id in sorted(zip(scores, unknown_ids)):
        f.write('%.3f\t%s\n' % (score, id))

    # t-SNE
    if do_tsne and tsne_file:
        tic = time.time()
        all_data = np.vstack((unknown_kmer_count, positive_kmer_count, negative_kmer_count))
        tsne_data = tsne.tsne(all_data, no_dims=2, perplexity=perplexity)
        del all_data
        tsne_time = time.time() - tic
        logger.info("t-SNE complete:  %d h %d m %.1f s" % (tsne_time//3600,(tsne_time%3600)//60,tsne_time%60))

        tsne_header = "t-SNE output (perplexity=%.1f)" % perplexity
        tsne_header += "\nunknown,positive,negative=(%d,%d,%d)\n" % (num_unknown, num_positive, num_negative)
        tsne_header += generate_summary(args).strip()
        tsne_header += "id,x,y"

        ids = np.concatenate((unknown_ids, positive_ids, negative_ids))
        save_tsne_data(tsne_file, tsne_data, ids, header=tsne_header)

    elif tsne_file and os.path.isfile(tsne_file):
        tsne_data = read_tsne_file(tsne_file)

    # Plotting
    if args.plot:
        logger.info("Creating plots...")
        tsne_unknown, tsne_positive, tsne_negative = chop([tsne_data, num_unknown, num_positive, num_negative])
        make_plots(tsne_positive, tsne_negative, tsne_unknown, filename=os.path.join(output_dir, 'tsne_plot_all.svg'))
        logger.info("Done creating plots.")
