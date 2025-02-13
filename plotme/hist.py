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

def plot_hist(data_fh, target, label_col, value_col, title, x_label, y_label, fig_width, fig_height, fontsize, bins, y_log, stacked, normalise, max_x):
  '''
  '''
  logging.info('starting...')

  import matplotlib.style
  try:
    matplotlib.style.use('seaborn-v0_8')
  except:
    matplotlib.style.use('seaborn-v0_8')

  included = total = 0
  results = collections.defaultdict(list)
  max_val = 0.0
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      if max_x is not None:
        to_add = min(max_x, float(row[value_col]))
      else:
        to_add = float(row[value_col])
      if label_col is None:
        results['data'].append(to_add)
      else:
        results[row[label_col]].append(to_add)
        
      max_val = max(max_val, to_add)
    except:
      logging.debug('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records with max_zval %.2f', included, total, max_val)

  if len(results) == 0:
    logging.warning('No data to plot')
    return

  fig = plt.figure(figsize=(fig_width, fig_height))
  rcParams.update({'font.size': fontsize})
  ax = fig.add_subplot(111)

  logging.debug('keys are %s', results.keys())
  logging.debug('stacked is %s', stacked)
    #plt.hist([ads, ads_nopass], label=('PASS', 'No PASS'), bins=int(max(ads + ads_nopass) * 100), stacked=False)
  if bins is not None:
    plt.hist([results[key] for key in sorted(results)], label=[key for key in sorted(results)], bins=bins, stacked=stacked, density=normalise)
  else:
    plt.hist([results[key] for key in sorted(results)], label=[key for key in sorted(results)], stacked=stacked, density=normalise)

  # this does overlap
  #else:
  #  for key in sorted(results.keys()):
  #    if bins is None:
  #      ax.hist(results[key], label=key, alpha=0.4, stacked=stacked)
  #    else:
  #      ax.hist(results[key], label=key, alpha=0.4, bins=int(bins), stacked=stacked)
  #plt.hist([ads, ads_nopass], label=('PASS', 'No PASS'), bins=int(max(ads + ads_nopass) * 100), stacked=False)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  if y_label is not None:
    ax.set_ylabel(y_label)
  if x_label is not None:
    ax.set_xlabel(x_label)
  ax.set_title(title)
  #ax.legend(loc='upper right')

  if y_log:
    ax.set_yscale('log', nonposy='clip')

  # place legend at right based on https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box/10154763#10154763
  handles, labels = ax.get_legend_handles_labels()
  lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01,1.0), borderaxespad=0)
  lgd.get_frame().set_edgecolor('#000000')

  logging.info('done processing %i of %i', included, total)
  plt.tight_layout()
  plt.savefig(target)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot a histogram')
  parser.add_argument('--label', required=False, help='x column name')
  parser.add_argument('--value', required=True, help='y column name')
  parser.add_argument('--title', required=False, help='graph title')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--bins', required=False, type=int, help='number of bins')
  parser.add_argument('--y_log', action='store_true', help='log scale on y axis')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--stacked', action='store_true', help='stack y values')
  parser.add_argument('--normalise', action='store_true', help='normalise y values as a density')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--height', required=False, type=float, default=8, help='height of plot')
  parser.add_argument('--width', required=False, type=float, default=12, help='width of plot')
  parser.add_argument('--fontsize', required=False, type=float, default=8, help='font size')
  parser.add_argument('--max', required=False, type=float, help='max x-value')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_hist(sys.stdin, args.target, args.label, args.value, args.title, args.x_label, args.y_label, args.width, args.height, args.fontsize, args.bins, args.y_log, args.stacked, args.normalise, args.max)

