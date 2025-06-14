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

import plotme.settings

def plot_heat(data_fh, target, xlabel, ylabel, zlabel, textlabel=None, width=12, height=18, fontsize=18, log=False, title=None, cmap=None, text_switch=0.5, x_label=None, y_label=None, is_numeric=False, x_map=None, y_map=None, x_order=None, y_order=None, x_highlight=None, colorbar_label=None, transparent=False, x_rotation=None, dpi=300, z_categorical=False, no_annotate=False, hide_colorbar=False, aspect=None, grid=False, x_order_no_sort=False, z_format='{:.2f}', y_order_no_sort=False):
  logging.info('starting...')

  included = total = 0
  results = {}
  text = {}
  xvals = set()
  xvals_order = []
  yvals = set()
  yvals_order = []
  max_zval = 0.0

  xmap = {}
  if x_map is not None:
    for item in x_map:
      provided, actual = item.split('=')
      actual = actual.replace('_', '\n')
      logging.debug(actual)
      xmap[provided] = actual

  ymap = {}
  if y_map is not None:
    for item in y_map:
      provided, actual = item.split('=')
      ymap[provided] = actual

  zmap = {}
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      if is_numeric:
        xval = float(row[xlabel]) # x axis value
        yval = float(row[ylabel]) # y axis value
      else:
        xval = row[xlabel] # x axis value
        yval = row[ylabel] # y axis value

      # use provided labels if provided
      if xval in xmap:
        xval = xmap[xval]
      elif len(xmap) > 0:
        logging.debug('xval %s not in xmap %s', xval, xmap)
      if yval in ymap:
        yval = ymap[yval]
      elif len(ymap) > 0:
        logging.debug('yval %s not in ymap %s', yval, ymap)

      if xval not in xvals:
        xvals_order.append(xval)
      if yval not in yvals:
        yvals_order.append(yval)
      xvals.add(xval)
      yvals.add(yval)
      if z_categorical:
        if row[zlabel] not in zmap:
          zmap[row[zlabel]] = len(zmap)
        zval = zmap[row[zlabel]]
      elif row[zlabel] == '':
        continue
      elif log:
        zval = math.log(float(row[zlabel]) + 1.0)
      else:
        zval = float(row[zlabel])
      max_zval = max(max_zval, zval)
      results['{}|{}'.format(xval, yval)] = zval

      if textlabel is None:
        logging.debug('zval %s -> %s', zval, z_format.format(zval))
        text['{}|{}'.format(xval, yval)] = z_format.format(zval)
      else:
        text['{}|{}'.format(xval, yval)] = row[textlabel].replace('/', '\n') # slashes for multiple lines
        
    except:
      logging.warn('Failed to include (is %s numeric?) %s', zlabel, row)
      raise

    total += 1

  logging.info('finished reading %i of %i records with max_zval %.2f', included, total, max_zval)
  logging.debug('results %s', results)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  if x_order is not None and len(x_order) > 0:
    #xvals = [x.replace('_', '\n') for x in x_order if x in xvals]
    xvals = [x for x in x_order if x in xvals]
  elif x_order_no_sort:
    xvals = xvals_order
  else:
    xvals = sorted(list(xvals))

  if y_order is not None and len(y_order) > 0:
    yvals = [y for y in y_order if y in yvals]
  else:
    if y_order_no_sort:
      yvals = yvals_order
    elif is_numeric:
      yvals = sorted(list(yvals))[::-1] # bottom left
    else:
      yvals = sorted(list(yvals))

  zvals = []
  tvals = []

  logging.info('evaluating %i yvals, starting with %s...', len(yvals), yvals[:5])
  for i, y in enumerate(yvals):
    zrow = []
    trow = []
    logging.info('evaluating %i xvals, starting with %s...', len(yvals), xvals[:5])
    for x in xvals:
      key = '{}|{}'.format(x, y)
      if key in results:
        zrow.append(results[key])
        trow.append(text[key])
      else:
        zrow.append(0.0)
        trow.append('')
    zvals.append(zrow)
    tvals.append(trow)
    if i % 10 == 0:
      logging.debug('added %i', i)

  matplotlib.rcParams.update({'font.size': fontsize})
  fig = plt.figure(figsize=(width, height))
  ax = fig.add_subplot(111)
  if grid:
    ax.set_xticks(np.arange(len(zvals[0]))-0.5, minor=True)
    ax.set_yticks(np.arange(len(zvals))-0.5, minor=True)
    ax.grid(which='minor', axis='both', linestyle='-')

  if cmap is None:
    im = ax.imshow(zvals, aspect=aspect)
  else:
    im = ax.imshow(zvals, cmap=cmap, aspect=aspect)

  if not hide_colorbar:
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.5)
    if colorbar_label is None:
      colorbar_label = zlabel
    cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

  ax.set_xticks(range(len(xvals)))
  ax.set_yticks(range(len(yvals)))
  ax.set_xticklabels(xvals, rotation=x_rotation) #, linespacing=2.0)
  if x_highlight is not None:
    for idx, xval in enumerate(xvals):
      if xval in x_highlight:
        ax.get_xticklabels()[idx].set_weight("bold")
  ax.set_yticklabels(yvals)

  if y_label is None:
    ax.set_ylabel(ylabel)
  else:
    ax.set_ylabel(y_label)

  if x_label is None:
    ax.set_xlabel(xlabel)
  else:
    ax.set_xlabel(x_label)

  if not no_annotate:
    logging.info('annotating %i yvals', len(yvals))
    for y in range(len(yvals)):
      for x in range(len(xvals)):
        #logging.info('%s vs %s', zvals[y][x], max_zval * text_switch)
        if zvals[y][x] > max_zval * text_switch:
          if cmap is not None and '_r' in cmap:
            text = ax.text(x, y, tvals[y][x], ha="center", va="center", color="w")
          else:
            text = ax.text(x, y, tvals[y][x], ha="center", va="center", color="k")
        else:
          if cmap is not None and '_r' in cmap:
            text = ax.text(x, y, tvals[y][x], ha="center", va="center", color="k")
          else:
            text = ax.text(x, y, tvals[y][x], ha="center", va="center", color="w")
      if y % 10 == 0:
        logging.debug('added %i', y)

  if title is None:
    ax.set_title('{} given {} and {}'.format(zlabel, xlabel, ylabel))
  else:
    ax.set_title(title)

  logging.info('done processing %i of %i', included, total)
  plt.tight_layout()
  plt.savefig(target, dpi=dpi, transparent=transparent)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot changes in signature')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--z', required=True, help='z column name')
  parser.add_argument('--z_format', required=False, default='{:.2f}', help='z numerical formatting')
  parser.add_argument('--z_categorical', action='store_true', help='z is categorical')
  parser.add_argument('--no_annotation', action='store_true', help='do not annotate cells')
  parser.add_argument('--text', required=False, help='text if different to z')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--cmap', required=False, help='cmap name')
  parser.add_argument('--colorbar_label', required=False, help='label on colorbar')
  parser.add_argument('--width', required=False, default=12, type=float, help='figsize width')
  parser.add_argument('--height', required=False, default=18, type=float, help='figsize height')
  parser.add_argument('--fontsize', required=False, default=18, type=int, help='fontsize')
  parser.add_argument('--dpi', required=False, default=300, type=int, help='dpi')
  parser.add_argument('--text_switch', required=False, default=0.5, type=float, help='where to change text colour')
  parser.add_argument('--log', action='store_true', help='log z')
  parser.add_argument('--transparent', action='store_true', help='transparent')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--is_numeric', action='store_true', help='axis are numeric')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--x_map', required=False, nargs='*', help='provided label=actual label')
  parser.add_argument('--y_map', required=False, nargs='*', help='provided label=actual label')
  parser.add_argument('--x_order', required=False, nargs='*', help='actual1 actual2...')
  parser.add_argument('--x_order_no_sort', action='store_true', help='do not sort xvals')
  parser.add_argument('--y_order', required=False, nargs='*', help='actual1 actual2...')
  parser.add_argument('--y_order_no_sort', action='store_true', help='actual1 actual2...')
  parser.add_argument('--x_highlight', required=False, nargs='*', help='xval1 xval2...')
  parser.add_argument('--x_rotation', required=False, help='vertical to change x label orientation')
  parser.add_argument('--hide_colorbar', action='store_true',  help='hide colorbar')
  parser.add_argument('--aspect', required=False, help='imshow aspect')
  parser.add_argument('--grid', action='store_true', help='imshow gridlines')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_heat(sys.stdin, args.target, args.x, args.y, args.z, args.text, args.width, args.height, args.fontsize, args.log, args.title, args.cmap, args.text_switch, args.x_label, args.y_label, args.is_numeric, args.x_map, args.y_map, args.x_order, args.y_order, args.x_highlight, args.colorbar_label, args.transparent, args.x_rotation, args.dpi, args.z_categorical, args.no_annotation, args.hide_colorbar, args.aspect, args.grid, args.x_order_no_sort, args.z_format, args.y_order_no_sort)
