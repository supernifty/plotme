#!/usr/bin/env python

import argparse
import collections
import csv
import logging
import math
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

def plot_box(data_fh, target, xlabel, ylabel, zlabel, title, x_label, y_label, x_order, y_order):
  '''
    xlabel: groups on x axis
    ylabel: colours
  '''
  logging.info('starting...')

  import matplotlib.style
  matplotlib.style.use('seaborn')

  included = total = 0
  results = collections.defaultdict(list)
  xvals = set()
  yvals = set()
  max_zval = 0.0
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      xval = row[xlabel] # group axis name
      yval = row[ylabel] # sub-group axis name
      xvals.add(xval)
      yvals.add(yval)
      zval = float(row[zlabel]) # value
      max_zval = max(max_zval, zval)
      results['{},{}'.format(xval, yval)].append(zval)
    except:
      logging.debug('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records with max_zval %.2f', included, total, max_zval)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  if x_order is None:
    xvals = sorted(list(xvals)) # groups
  else:
    xvals = x_order # groups
  if y_order is None:
    yvals = sorted(list(yvals)) # sub-groups
  else:
    yvals = y_order

  logging.debug('xvals %s yvals %s', xvals, yvals)

  #fig, ax = plt.subplots()

  fig_width = min(18, max(6, len(xvals) * len(yvals)))
  fig = plt.figure(figsize=(fig_width, fig_width * 0.7))
  ax = fig.add_subplot(111)

  width = 6 / fig_width
  ind = 1 + np.arange(len(xvals)) * fig_width / len(xvals)  # the x locations for the groups
  logging.debug('ind is %s, width is %f fig_width is %f', ind, width, fig_width)

  boxes = []
  for idx in range(len(yvals)):
    offset = idx * width - (len(yvals) - 1) * width / 2
    vals = [results['{},{}'.format(x, yvals[idx])] for x in xvals]
    logging.debug('adding values %s for %s at %s %s', vals, yvals[idx], ind, offset)
    #rects = ax.bar(ind + offset, vals, width, label=yvals[idx]) 
    for c, val in enumerate(vals):
      rects = ax.boxplot(val, notch=0, sym='+', vert=1, whis=1.5, positions=[ind[c] + offset], widths=width * 0.9, patch_artist=True, boxprops=dict(facecolor="C{}".format(idx)), medianprops=dict(color='#000000'))
      boxes.append(rects)
    #for rect in rects:
    #  height = rect.get_height()
    #  ax.annotate('{}'.format(height),
    #    xy=(rect.get_x() + rect.get_width() / 2, height),
    #    xytext=(0, 3),  # use 3 points offset
    #    textcoords="offset points",  # in both directions
    #    ha='center', va='bottom')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  if y_label is not None:
    ax.set_ylabel(y_label)
  if x_label is not None:
    ax.set_xlabel(x_label)
  ax.set_title(title)
  ax.set_xticks(ind)
  ax.set_xticklabels(xvals)
  ax.set_xlim((-1, max(ind) + 1 + width))

  # place legend at right based on https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box/10154763#10154763
  #handles, labels = ax.get_legend_handles_labels()
  #handles = [boxes[0]["boxes"][0], boxes[2]["boxes"][0]]
  handles = [boxes[c]["boxes"][0] for c in range(0, len(boxes), len(xvals))]
  labels = yvals
  lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01,1.0), borderaxespad=0)
  lgd.get_frame().set_edgecolor('#000000')

  #fig = plt.figure(figsize=(figsize, 1 + int(figsize * len(yvals) / len(xvals))))
  #ax = fig.add_subplot(111)

  logging.info('done processing %i of %i', included, total)
  plt.tight_layout()
  plt.savefig(target)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot a bar chart')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--z', required=True, help='z column name')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--x_order', required=False, nargs='*', help='order of x axis')
  parser.add_argument('--y_order', required=False, nargs='*', help='order of y axis')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_box(sys.stdin, args.target, args.x, args.y, args.z, args.title, args.x_label, args.y_label, args.x_order, args.y_order)
