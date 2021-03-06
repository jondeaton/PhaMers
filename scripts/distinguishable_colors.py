#!/usr/bin/env python2.7
"""
Distinguishable Colors module
Jonathan Deaton, 05.09.16

This module is used to generate distinguishable colors. This module implements functions that return colors
from the Colour Alphabet Project's color "letters" by default, or can optionally generates colors over specified ranges
of hues. For this reason, it is recommended to generate fewer than 26 colors.

=== Usage in Python ===
    import distinguishable_colors as dc
    colors = dc.get_colors(num_colors)
    colors = dc.get_colors(num_colors, c=['red','blue','green'])

=== Usage from command line for testing ===
    python distinguishable_colors.py 6 -c "red green blue"
"""

import numpy as np
import colorsys
import random

# This is so that we get the same colors every time
random.seed(11)

__version__ = 1.0
__author__ = "Jonathan Deaton (jdeaton@stanford.edu)"
__license__ = "No license"

default_colors = ['red', 'orange', 'green', 'blue', 'purple']


def get_colors(n_colors, c=default_colors):
    """
    A function for generating a number of colors that are able distinguishable by signt. If the number of colors is
    less than 26, the numbers will be selected from the 2010 Colour Alphabet Project color "letters", and if the number\
    is greater than 26, then the colors will be generated by even distribution over the spectrum of hues.
    :param n_colors: The number of colors to generate
    :param c: A list of strings specifying the names of colors to use
    :return: A n_colors by 3 numpy array with rows specifying RGB colors
    """

    if n_colors <= 26 and c == default_colors:
        # Default Colour Alphabet Project "letters"
        return np.array(random.sample(color_alphabet(), n_colors)) / 255
    elif n_colors <= 100 and c == default_colors:
        return np.array(random.sample(medium_colors(), n_colors)) / 255
    elif n_colors > 100 and c == default_colors:
        first_100_colors = np.array(random.sample(medium_colors(), 100)) / 255
        remaining_colors = get_colors(n_colors - 100, c=default_colors[::-1])
        return np.vstack((first_100_colors, remaining_colors))

    color_dict = dict(zip(default_colors, [(-20, 25), (25, 70), (70, 170), (170,270), (270, 340)]))

    if isinstance(c, str):
        c = [c]

    shades_per_color = 1 + n_colors // len(c)
    hues = np.zeros((len(c), shades_per_color))
    for i in xrange(len(c)):
        color = c[i]
        range = np.array(color_dict[color]) / 360.0
        shades = np.linspace(range[0], range[1], 2 + shades_per_color)[1:-1] % 1
        hues[i, :] = shades

    H = np.sort(np.transpose(hues).reshape((1, hues.shape[0] * hues.shape[1]))[0][:n_colors])
    R, T = 0.3, 0.3
    S = (1 - R) + R * ((1 + np.arange(n_colors)) % 2)
    L = 0.5 + T * (np.arange(n_colors) % 3) / 2

    X = np.transpose(np.vstack((H, L, S)))
    colors = []
    for h, l, s in X:
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append(np.array([r, g, b]))

    colors = np.array(colors)
    np.random.shuffle(colors)

    return colors


def get_color_dict(list_of_things, c=default_colors):
    """
    This function makes a dictionary that maps elements of a list to colors
    :param list_of_things: A list of things each of which will be assigned a color
    :return: A dictionary that maps thing --> color
    """
    return dict(zip(list_of_things, get_colors(len(list_of_things), c=c)))


def test(n_colors, c=default_colors):
    """
    This function is for testing the colors generated by get_colors(). This function will use matplotlib to make
    an image displaying each of the colors that would be generated by a call to get_colors with the same arguments
    :param n_colors: The number of colors to generate and use
    :param c: A list of strings specifying the names of colors to use
    :return: None
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    X, Y = fig.get_dpi() * fig.get_size_inches()
    colors = get_colors(n_colors, c=c)
    ncols = np.ceil(np.sqrt(n_colors))
    nrows = 1 + n_colors / ncols
    # row height
    h = Y / (nrows + 1)
    # col width
    w = X / ncols
    for i in xrange(n_colors):
        color = colors[i]
        name = str(i + 1)
        row = i / nrows
        col = i % ncols
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, y, name, fontsize=(h * 0.8), horizontalalignment='left', verticalalignment='center')

        # Add extra black line a little bit thicker to make
        # clear colors more visible.
        ax.hlines(y, xi_line, xf_line, color='black', linewidth=(h * 0.7))
        ax.hlines(y + h * 0.1, xi_line, xf_line, color=color, linewidth=(h * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.show()


def get_color_alphabet(address='https://en.wikipedia.org/wiki/Help:Distinguishable_colors'):
    """
    A function for retrieving the 2010 Colour Alphabet Project color "letters" from a wikipedia page
    :param address: Optional argument for the address of the wikipedia page where these letters are found
    :return: A 26 by 3 numpy array of RGB colors on the 0-255 color scale
    """
    import urllib2
    content = urllib2.urlopen(address).read()
    color_lines = [line for line in content.split('\n') if '<td>' in line and '{' in line and '}' in line]
    colors = np.array([])
    for color_line in color_lines:
        color = np.array(color_line[1+color_line.index('{'):color_line.index('}')].split(',')).astype(int)
        if len(color) == 3:
            if len(colors) == 0:
                colors = np.array(color).astype(int)
            else:
                colors = np.vstack((colors, color))
    return colors


def color_alphabet(again=False):
    """
    This function has the 2010 Colour Alphabet Project color "letters" preloaded, and will return them when called.
    :param again: This optional parameter specifies if the colors should be retrieved from the web once again
    :return: The 26 Colour Alphabet Project color "letters", in 0-255 RBG format in a 26 by 3 numpy array
    """
    if again:
        try:
            colors = get_color_alphabet()
        except:
            print "Wasn't able to get info from web!"
            colors = None
    else:
        colors = np.array([[240, 163, 255], [0, 117, 220], [153, 63, 0], [76, 0, 92], [25, 25, 25], [0, 92, 49], [43, 206, 72], [255, 204, 153], [128, 128, 128], [148, 255, 181], [143, 124, 0], [157, 204, 0], [194, 0, 136], [0, 51, 128], [255, 164, 5], [255, 168, 187], [66, 102, 0], [255, 0, 16], [94, 241, 242], [0, 153, 143], [224, 255, 102], [116, 10, 255], [153, 0, 0], [255, 255, 128], [255, 255, 0], [255, 80, 5]], dtype=float)
    return colors


def medium_colors():
    """
    This function gives you a bunch of colors that all look different
    :return: A (?) by 3 numpy array describing RGB colors
    """
    colors = np.array([[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,0,255],[0,255,255],[255,255,0],[0,0,0],
                       [112,219,147],[92,51,23],[159,95,159],[181,166,66],[217,217,25],[166,42,42],[140,120,83],
                       [166,125,61],[95,159,159],[217,135,25],[184,115,51],[255,127,0],[66,66,111],[92,64,51],
                       [47,79,47],[74,118,110],[79,79,47],[153,50,205],[135,31,120],[107,35,142],[47,79,79],
                       [151,105,79],[112,147,219],[133,94,66],[84,84,84],[133,99,99],[209,146,117],[142,35,35],
                       [245,204,176],[35,142,35],[205,127,50],[219,219,112],[192,192,192],[82,127,118],[147,219,112],
                       [33,94,33],[78,47,47],[159,159,95],[192,217,217],[168,168,168],[143,143,189],[233,194,166],
                       [50,205,50],[228,120,51],[142,35,107],[50,205,153],[50,50,205],[107,142,35],[234,234,174],
                       [147,112,219],[66,111,66],[127,0,255],[127,255,0],[112,219,219],[219,112,147],[166,128,100],
                       [47,47,79],[35,35,142],[77,77,255],[255,110,199],[0,0,156],[235,199,158],[207,181,59],
                       [255,127,0],[255,36,0],[219,112,219],[143,188,143],[188,143,143],[234,173,234],[217,217,243],
                       [89,89,171],[111,66,66],[140,23,23],[35,142,104],[107,66,38],[142,107,35],[230,232,250],
                       [50,153,204],[0,127,255],[255,28,174],[0,255,127],[35,107,142],[56,176,222],[219,147,112],
                       [216,191,216],[173,234,234],[92,64,51],[205,205,205],[79,47,79],[204,50,153],[216,216,191],
                       [153,204,50]], dtype=float)

    return colors


def get_more_colors(address='http://www.two4u.com/color/medium-txt.html'):
    """
    This function gets 100 colors from a webcie that I found
    Copyright 1995-1999 Mark Koenen: color@two4u.com
    :param address: the addres of the webcite
    :return: The RGB colors in a numpy array
    """
    import urllib2
    lines = urllib2.urlopen(address).read().split('\n')
    colors = np.array([line.split()[-3:] for line in lines if 'cgi-bin/color' in line]).astype(float) / 255
    return colors

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_colors', type=int,help='Number of colors')
    parser.add_argument('-c', '--colors', default=' '.join(default_colors), type=str, help='Colors to use')
    args = parser.parse_args()

    num_colors = args.num_colors
    colors = args.colors.split(' ')

    test(num_colors, c=colors)
