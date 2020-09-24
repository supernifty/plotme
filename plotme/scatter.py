#!/usr/bin/env python

import argparse
import csv
import logging
import math
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

import plotme.settings

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS = ('o', 'x', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '.', ',', '|', '_')

def plot_scatter(data_fh, target, xlabel, ylabel, zlabel, figsize, fontsize, log, title, x_label, y_label, wiggle, delimiter, z_color, z_color_map, label, join):
  logging.info('starting...')
  matplotlib.style.use('seaborn')

  included = total = 0
  xvals = []
  yvals = []
  zvals = []
  cvals = []
  mvals = []
  lvals = []

  zvals_seen = []
  markers_seen = set()
  colors_seen = set()

  for row in csv.DictReader(data_fh, delimiter=delimiter):
    try:
      included += 1
      xval = float(row[xlabel]) + (random.random() - 0.5) * 2 * wiggle # x axis value
      yval = float(row[ylabel]) + (random.random() - 0.5) * 2 * wiggle # y axis value
      xvals.append(xval)
      yvals.append(yval)
      if zlabel is not None:
        if row[zlabel] not in zvals_seen:
          zvals_seen.append(row[zlabel])

        z_color_map_found = False
        if z_color_map is not None:
          for m in z_color_map:
            name, value = m.split(':')
            if name == row[zlabel]:
              color, marker = value.split('/')
              cvals.append(color)
              colors_seen.add(color)
              mvals.append(marker)
              markers_seen.add(marker)
              z_color_map_found = True
              break

        if z_color and not z_color_map_found:
          ix = zvals_seen.index(row[zlabel])
          cvals.append(COLORS[ix % len(COLORS)])
          colors_seen.add(COLORS[ix % len(COLORS)])
          jx = int(ix / len(COLORS))
          mvals.append(MARKERS[jx % len(MARKERS)])
          markers_seen.add(MARKERS[jx % len(MARKERS)])

        zvals.append(row[zlabel])

      if label is not None:
        lvals.append(row[label])

    except:
      logging.warn('Failed to include (is %s numeric?) %s', zlabel, row)
      raise

    total += 1

  logging.info('finished reading %i of %i records', included, total)

  if len(xvals) == 0:
    logging.warn('No data to plot')
    return

  matplotlib.rcParams.update({'font.size': fontsize})
  fig = plt.figure(figsize=(figsize, 1 + int(figsize * len(yvals) / len(xvals))))
  ax = fig.add_subplot(111)

  if y_label is None:
    ax.set_ylabel(ylabel)
  else:
    ax.set_ylabel(y_label)

  if x_label is None:
    ax.set_xlabel(xlabel)
  else:
    ax.set_xlabel(x_label)

  if z_color or z_color_map is not None:
    for zval in zvals_seen:
      logging.debug('zval %s...', zval)
      vals = [list(x) for x in zip(xvals, yvals, zvals, cvals, mvals) if x[2] == zval]
      ax.scatter([x[0] for x in vals], [x[1] for x in vals], c=[x[3] for x in vals], marker=vals[0][4], label=zval, alpha=0.8)
      ax.legend()
      if join: # TODO does this work?
        ax.join([x[0] for x in vals], [x[1] for x in vals], c=[x[3] for x in vals], marker=vals[0][4], label=zval, alpha=0.8)
  else:
    ax.scatter(xvals, yvals)
    if join:
      ax.plot(xvals, yvals)

  if zlabel is not None:
    if not z_color:
      for x, y, z in zip(xvals, yvals, zvals):
        ax.annotate(z, (x, y), fontsize=fontsize)

  # alternative labelling
  if label is not None:
    for x, y, z in zip(xvals, yvals, lvals):
      ax.annotate(z, (x, y), fontsize=fontsize)

  if title is not None:
    ax.set_title(title)

  logging.info('done processing %i of %i. saving to %s...', included, total, target)
  plt.tight_layout()
  plt.savefig(target, dpi=plotme.settings.DPI, transparent=False) #plotme.settings.TRANSPARENT)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Scatter plot')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--z', required=False, help='z column name')
  parser.add_argument('--label', required=False, help='label column')
  parser.add_argument('--z_color', action='store_true', help='use colours for z')
  parser.add_argument('--z_color_map', required=False, nargs='+', help='specify color/marker for z: label=color:marker')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--figsize', required=False, default=12, type=float, help='figsize width')
  parser.add_argument('--fontsize', required=False, default=18, type=int, help='fontsize')
  parser.add_argument('--wiggle', required=False, default=0, type=float, help='randomly perturb data')
  parser.add_argument('--delimiter', required=False, default='\t', help='input file delimiter')
  parser.add_argument('--log', action='store_true', help='log z')
  parser.add_argument('--join', action='store_true', help='join points')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_scatter(sys.stdin, args.target, args.x, args.y, args.z, args.figsize, args.fontsize, args.log, args.title, args.x_label, args.y_label, args.wiggle, args.delimiter, args.z_color, args.z_color_map, args.label, args.join)
