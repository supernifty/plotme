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

COLORS=['#ffa600', '#55a868', '#2f4b7c', '#a05195', '#003f5c', '#665191', '#ff7c43', '#f95d6a', '#d45087']

#  plot_box(sys.stdin, args.target, args.x, args.y, args.z, args.title, args.x_label, args.y_label, args.x_order, args.y_order, args.width, args.height, args.fontsize, args.x_label_rotation, args.no_legend, args.linewidth, args.dpi)
def plot_seg(data_fh, target, xlabel, ylabel, lower, mean, upper, title, x_label, y_label, x_order, y_order, fig_width, fig_height, fontsize, x_label_rotation='vertical', no_legend=False, linewidth=1, dpi=300, separator=False):
  '''
    xlabel: category
    ylabel: sub-category
    zlabel: 
  '''
  logging.info('starting...')

  import matplotlib.style
  try:
    matplotlib.style.use('seaborn-v0_8')
  except:
    matplotlib.style.use('seaborn-v0_8')
  rcParams.update({'lines.markeredgewidth': 0.1}) # seaborn removes fliers

  included = total = 0
  results = collections.defaultdict(dict)
  xvals = set()
  yvals = set()
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      xval = row[xlabel] # group axis name
      yval = row[ylabel] # sub-group axis name
      results[xval][yval] = (float(row[mean]), float(row[lower]), float(row[upper])) 
      xvals.add(xval)
      yvals.add(yval)
    except:
      logging.warn('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records', included, total)
  logging.debug('xvals %s yvals %s results %s', xvals, yvals, results)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  logging.debug('xvals %s yvals %s', xvals, yvals)

  #fig, ax = plt.subplots()

  #fig_width = min(18, max(6, len(xvals) * len(yvals)))
  #fig = plt.figure(figsize=(fig_width, fig_width * 0.7))
  rcParams.update({'font.size': fontsize})
  fig = plt.figure(figsize=(fig_width, fig_height))
  plt.rc('legend',fontsize=fontsize)
  ax = fig.add_subplot(111)
  ax.tick_params(axis='x', labelsize=fontsize)
  ax.tick_params(axis='y', labelsize=fontsize)
  ax.grid(axis='y', linewidth=0) # no lines on y-axis

  #ax.errorbar(list(xvals), , xerr=errs, fmt='o')
  # do each sub-category with a different color
  for ydx, yval in enumerate(sorted(yvals)):
    xs = []
    ys = []
    ls = []
    us = []
    for xdx, xval in enumerate(sorted(xvals)):
      logging.debug('processing %s %s...', yval, xval)
      position = xdx * len(yvals) + ydx
      xs.append(position)
      ys.append(results[xval][yval][0])
      ls.append(results[xval][yval][0] - results[xval][yval][1])
      us.append(results[xval][yval][2] - results[xval][yval][0])
    logging.debug('xs %s ys %s for %i', xs, ys, ydx)
    ax.errorbar(ys, xs, xerr=[ls, us], fmt='o', c=COLORS[ydx], label=yval)

  if separator:
    for x in range(len(xvals)-1):
      ax.axhline((x + 1) * len(yvals) - 0.5, color='white', linewidth=1)

  labels = [xval.replace('/', '\n') for xval in sorted(xvals)]
  positions = [x * len(yvals) + ((len(yvals)-1) / 2) for x in range(len(xvals))]

  ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(positions))
  ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
  ax.legend()
  
  ax.set_title(title)

  #if separator:
  #  for i, x in enumerate(ind[:-1]):
  #    ax.axvline((x + ind[i+1]) / 2, color='white', linewidth=1)
  #    logging.debug('vline at %f', x)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  if y_label is not None:
    ax.set_ylabel(y_label, fontsize=fontsize)
  if x_label is not None:
    ax.set_xlabel(x_label, fontsize=fontsize)
  ax.set_title(title, fontsize=fontsize)

  logging.info('done processing %i of %i. plot at dpi %i', included, total, dpi)
  plt.tight_layout()
  plt.savefig(target, dpi=dpi)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot a bar chart')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--lower', required=True, help='z column name')
  parser.add_argument('--mean', required=True, help='z column name')
  parser.add_argument('--upper', required=True, help='z column name')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--x_order', required=False, nargs='*', help='order of x axis')
  parser.add_argument('--y_order', required=False, nargs='*', help='order of y axis')
  parser.add_argument('--separator', action='store_true', help='separate groups')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--height', required=False, type=float, default=8, help='height of plot')
  parser.add_argument('--width', required=False, type=float, default=12, help='width of plot')
  parser.add_argument('--fontsize', required=False, type=int, default=8, help='font size')
  parser.add_argument('--x_label_rotation', required=False, default='vertical', help='rotation of x labels vertical or horizontal')
  parser.add_argument('--linewidth', default=1, type=float, help='line width')
  parser.add_argument('--dpi', default=300, type=float, help='dpi')
  parser.add_argument('--no_legend', action='store_true', help='no legend')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_seg(sys.stdin, args.target, args.x, args.y, args.lower, args.mean, args.upper, args.title, args.x_label, args.y_label, args.x_order, args.y_order, args.width, args.height, args.fontsize, args.x_label_rotation, args.no_legend, args.linewidth, args.dpi, args.separator)


