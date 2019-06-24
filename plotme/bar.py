#!/usr/bin/env python

import argparse
import csv
import logging
import math
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

def plot_bar(data_fh, target, xlabel, ylabel, zlabel, figsize, title, x_label, y_label):
  '''
    xlabel: groups on x axis
  '''
  logging.info('starting...')

  included = total = 0
  results = {}
  xvals = set()
  yvals = set()
  max_zval = 0.0
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      xval = row[xlabel] # group axis value
      yval = row[ylabel] # sub-group axis value
      xvals.add(xval)
      yvals.add(yval)
      zval = float(row[zlabel])
      max_zval = max(max_zval, zval)
      results['{},{}'.format(xval, yval)] = zval
    except:
      logging.debug('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records with max_zval %.2f', included, total, max_zval)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  xvals = sorted(list(xvals)) # groups
  yvals = sorted(list(yvals)) # sub-groups

  logging.debug('xvals %s yvals %s', xvals, yvals)

  #fig, ax = plt.subplots()

  fig_width = len(xvals) * len(yvals)
  fig = plt.figure(figsize=(fig_width, fig_width * 0.7))
  ax = fig.add_subplot(111)

  width = 0.7
  ind = np.arange(len(xvals)) * fig_width / len(xvals)  # the x locations for the groups
  logging.debug('ind is %s, width is %f', ind, width)

  for idx in range(len(yvals)):
    offset = idx * width - (len(yvals) - 1) * width / 2
    vals = [results['{},{}'.format(x, yvals[idx])] for x in xvals]
    logging.debug('adding values %s for %s at %s', vals, yvals[idx], ind + offset)
    rects = ax.bar(ind + offset, vals, width, label=yvals[idx]) 

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.set_xticks(ind)
  ax.set_xticklabels(xvals)
  ax.legend(loc='upper right')

  #fig = plt.figure(figsize=(figsize, 1 + int(figsize * len(yvals) / len(xvals))))
  #ax = fig.add_subplot(111)

  logging.info('done processing %i of %i', included, total)
  plt.tight_layout()
  plt.savefig(target)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot changes in signature')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--z', required=True, help='z column name')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--figsize', required=False, default=12, type=int, help='figsize width')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_bar(sys.stdin, args.target, args.x, args.y, args.z, args.figsize, args.title, args.x_label, args.y_label)

