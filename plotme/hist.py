#!/usr/bin/env python
'''
  histogram
  Label,Value
'''

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

def plot_hist(data_fh, target, label_col, value_col, title, x_label, y_label, fig_width, fig_height, fontsize):
  '''
  '''
  logging.info('starting...')

  import matplotlib.style
  matplotlib.style.use('seaborn')

  included = total = 0
  results = collections.defaultdict(list)
  max_val = 0.0
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      results[row[label_col]].append(float(row[value_col]))
      max_val = max(max_val, float(row[value_col]))
    except:
      logging.debug('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records with max_zval %.2f', included, total, max_val)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  fig = plt.figure(figsize=(fig_width, fig_height))
  rcParams.update({'font.size': fontsize})
  ax = fig.add_subplot(111)

  for key in sorted(results.keys()):
    ax.hist(results[key], label=key, alpha=0.5)
  #plt.hist([ads, ads_nopass], label=('PASS', 'No PASS'), bins=int(max(ads + ads_nopass) * 100), stacked=False)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  if y_label is not None:
    ax.set_ylabel(y_label)
  if x_label is not None:
    ax.set_xlabel(x_label)
  ax.set_title(title)
  #ax.legend(loc='upper right')

  # place legend at right based on https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box/10154763#10154763
  handles, labels = ax.get_legend_handles_labels()
  lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01,1.0), borderaxespad=0)
  lgd.get_frame().set_edgecolor('#000000')

  logging.info('done processing %i of %i', included, total)
  plt.tight_layout()
  plt.savefig(target)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot a bar chart')
  parser.add_argument('--label', required=True, help='x column name')
  parser.add_argument('--value', required=True, help='y column name')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--height', required=False, type=float, default=8, help='height of plot')
  parser.add_argument('--width', required=False, type=float, default=12, help='width of plot')
  parser.add_argument('--fontsize', required=False, type=float, default=8, help='font size')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_hist(sys.stdin, args.target, args.label, args.value, args.title, args.x_label, args.y_label, args.width, args.height, args.fontsize)

