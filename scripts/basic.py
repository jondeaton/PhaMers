#!/usr/bin/env python
'''
This script implements some basic function used in various places
'''

import os
import sys
import numpy as np
import logging

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_summary(args, line_start='# ', header=''):
    '''
    This function makes a summary for an output file from an argparse parsed arguments object
    The summary will
    :param args: The parsed arguments object used to parse the arguments from the function call
    :return: a beautiful summary
    '''
    str_rep = args.__str__()
    str_rep = str_rep.replace('Namespace(', line_start)
    str_rep = str_rep.replace(')', '')
    str_rep = str_rep.replace(', ', '\n' + line_start)
    str_rep = str_rep.replace('=', ':\t') + '\n'
    str_rep = line_start + header + '\n' + str_rep
    return str_rep


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    Created by Trent Mick on Fri, 19 Feb 2010 (MIT)
    Taken from: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


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

def decide_file(specified_file, found_file, abort=None):
    '''
    This function decides which file should be used
    :param specified_file: A file which was specified explicitly by the user
    :param found_file: A file which was found somewhere but not explicitly provided
    :param abort: Pass a string to this parameter to exit the program if neither file exists
    :return: The file which should be used (preference to provided_file), or None, if neither file exists.
    '''
    if specified_file is not None and os.path.exists(specified_file):
        # The explicitly provided file is legit... return it
        return specified_file
    elif specified_file is None and found_file is not None and os.path.exists(found_file):
        # Didn't provide a file explicitly but one was found
        return found_file
    elif abort:
        exit(abort)
    else:
        return None


def search_for_file(directory, start=None, contain=None, end=None):
    '''
    This function returns the path of a file within a directory that has a given ending
    :param directory: The directory to search in
    :param ending: The ending of the file of interest
    :return: A string with the full path of the file of interest
    '''
    all_files_in_dir = os.listdir(directory)
    search_results = [file for file in all_files_in_dir if
                       (not start or file.startswith(start)) and
                       (not contain or contain in file) and
                       (not end or file.endswith(end))]
    return [os.path.join(directory, file) for file in search_results]


def represents_int(string):
    '''
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    '''
    try:
        int(string)
        return True
    except ValueError:
        return False


def represents_float(string):
    '''
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    print "This is a module, not meant to be run from the command line."
