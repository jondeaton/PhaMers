#!/usr/bin/env python
"""
basic.py

This script implements some basic function used in various places
"""

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


def generate_summary(args, line_start='', header=''):
    """
    This function makes a summary for an output file from an argparse parsed arguments object
    The summary will
    :param args: The parsed arguments object used to parse the arguments from the function call
    :return: a beautiful summary
    """
    if args is None:
        return ""
    str_rep = args.__str__()
    str_rep = str_rep.replace('Namespace(', line_start)
    str_rep = str_rep.replace(')', '')
    str_rep = str_rep.replace(', ', '\n' + line_start)
    str_rep = str_rep.replace('=', ':\t') + '\n'
    str_rep = line_start + header + '\n' + str_rep
    return str_rep


def query_yes_no(prompt, default="yes"):
    
    yes = set(["yes", "y", "ye"])
    no = set(["no", "n"])
    exit_responses = set(["exit", "quit", "stop"])

    if default is True or default in yes:
        yes.add("")
        options_text = "Y/n"
    elif default is False or default in no:
        no.add("")
        options_text = "y/N"
    elif default is None:
        options_text = "y/n"
    else:
        raise ValueError

    stdout = "{prompt} [{options}] ".format(prompt=prompt, options=options_text) 

    response = "?"
    while True:
        response = raw_input(stdout).lower()
        if response in yes:
            return True
        elif response in no:
            return False
        elif response in exit_responses:
            exit()
        else:
            print "Answer with yes/no/exit."


def chop(array, chops):
    """
    This function is for separating the rows of a numpy array
    i.e. chop(X, [10, 15]) will return a list in which the first element is a numpy array containing the first 10 rows of X,
    and the second element is a numpy array containing next 15 rows of X.
    :param array: The array to chop up
    :param chops: A list of integers to chop the array into
    :return: A list of numpy arrays split as described previously
    """
    chopped = []
    if type(array) is not np.ndarray:
        array = np.array(array)
    at = 0
    for chop_size in chops:
        chopped.append(array[at: at + chop_size])
        at += chop_size
    return chopped

def decide_file(specified_file, found_file, abort=None):
    """
    This function decides which file should be used
    :param specified_file: A file which was specified explicitly by the user
    :param found_file: A file which was found somewhere but not explicitly provided
    :param abort: Pass a string to this parameter to exit the program if neither file exists
    :return: The file which should be used (preference to provided_file), or None, if neither file exists.
    """
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


def search_for_file(directory, start=None, contain=None, end=None, contains=None, first=False, recency=False, files_only=False, dirs_only=False):
    """
    This function returns the path of a file within a directory that has a given ending
    :param directory: The directory to search in
    :param start: Something that the file name must start with
    :param contain: Something that the file name must contain
    :param contains: Something that the file name mmust contain
    :param ending: The ending of the file of interest
    :param first: Return the first file found with these
    :param recency: Sort files by when they were last changed. Most recently changed files are at the end of the array.
    :return: A string with the full path of the file of interest
    """
    all_files_in_dir = os.listdir(directory)
    search_results = [file for file in all_files_in_dir if
                      (not start or file.startswith(start)) and
                      (not contain or contain in file) and
                      (not contains or contains in file) and
                      (not end or file.endswith(end))]

    files = [os.path.join(directory, file) for file in search_results]
    if files_only and not dirs_only:
        files = [file for file in files if not os.path.isdir(file)]
    if dirs_only and not files_only:
        files = [file for file in files if os.path.isdir(file)]
    if recency:
        files.sort(key=lambda x: os.path.getmtime(x))
    if first:
        if len(files) > 0:
            return files[0]
        else:
            return None
    else:
        return files


def represents_int(string):
    """
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def represents_float(string):
    """
    This function determines if a string represents an integer
    :param s: The string
    :return: True if the string represents an integer
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def list_mode(list):
    """
    This function returns the mathematical mode of a list
    :param list: A list of items
    :return: The item which occurs most often in the list
    """
    return max(set(list), key=list.count)


if __name__ == '__main__':
    print "This is a module, not meant to be run from the command line."

