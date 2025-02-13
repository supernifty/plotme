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

COLORS=['#003f5c', '#2f4b7c', '#ffa600', '#a05195', '#665191', '#ff7c43', '#f95d6a', '#d45087']

def plot_box(data_fh, target, xlabel, ylabel, zlabel, title, x_label, y_label, x_order, y_order, fig_width, fig_height, fontsize, significance, significance_nobar, separator, include_zero=False, x_label_rotation='vertical', y_log=False, annotate=None, annotate_location=None, include_other=None, violin=False, y_counts=False, color_index=0, colors=COLORS, y_max=None, sig_ends=0.01, colors_special=None, no_legend=False, sig_fontsize=8, markersize=6, linewidth=1, dpi=300, horizontal=False):
  '''
    xlabel: groups on x axis
    ylabel: colours
  '''
  logging.info('starting...')

  import matplotlib.style
  try:
    matplotlib.style.use('seaborn-v0_8')
  except:
    matplotlib.style.use('seaborn-v0_8')
  rcParams.update({'lines.markeredgewidth': 0.1}) # seaborn removes fliers

  included = total = 0
  results = collections.defaultdict(list)
  xvals = set()
  yvals_count = collections.defaultdict(int)
  max_zval = -1e99
  min_zval = 1e99
  for row in csv.DictReader(data_fh, delimiter='\t'):
    try:
      included += 1
      xval = row[xlabel] # group axis name
      if ylabel is None:
        yval = ''
      else:
        yval = row[ylabel] # sub-group axis name
      xvals.add(xval)
      yvals_count[yval] += 1
      zval = float(row[zlabel]) # value
      max_zval = max(max_zval, zval)
      min_zval = min(min_zval, zval)
      results['{},{}'.format(xval, yval)].append(zval)
    except:
      logging.debug('Failed to include %s', row)

    total += 1

  logging.info('finished reading %i of %i records with range %.2f to %.2f', included, total, min_zval, max_zval)

  if len(results) == 0:
    logging.warn('No data to plot')
    return

  if x_order is None:
    xvals = sorted(list(xvals)) # groups
  else:
    xvals = x_order # groups

  if y_order is None:
    yvals = sorted(list(yvals_count)) # sub-groups
  else:
    yvals = y_order

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
  ax.grid(axis='x', linewidth=0) # no lines on x-axis

  if y_log:
    ax.set_yscale('log', nonposy='clip')

  width = fig_width / len(xvals) / len(yvals) * 0.8 # max width of each bar
  ind = width * len(yvals) / 2 + np.arange(len(xvals)) * fig_width / len(xvals)  # the x locations for the groups
  logging.debug('ind is %s, width is %f fig_width is %f', ind, width, fig_width)

  if separator:
    for i, x in enumerate(ind[:-1]):
      ax.axvline((x + ind[i+1]) / 2, color='white', linewidth=1)
      logging.debug('vline at %f', x)

  special_count = 0
  boxes = []
  positions = []
  for idx in range(len(yvals)):
    offset = idx * width * 0.9 - (len(yvals) - 1) * width / 2 # offset of each bar compared to ind (x centre for each group)
    #offset = idx * width * 0.9 - (len(yvals) - 1) * width / 2 # offset of each bar compared to ind (x centre for each group)
    vals = [results['{},{}'.format(x, yvals[idx])] for x in xvals]
    logging.debug('adding values %s for %s at %s %s', vals, yvals[idx], ind, offset)
    if len(yvals) > 6:
      if idx < len(colors):
        color = colors[(idx + color_index) % len(colors)]
      else:
        color = 'C{}'.format((idx + color_index) - len(colors))
    else:
      color = 'C{}'.format(idx + color_index)
    #rects = ax.bar(ind + offset, vals, width, label=yvals[idx]) 
    for c, val in enumerate(vals):
      if colors_special is not None:
        color = colors_special[special_count % len(colors_special)]
        special_count += 1
      position = [ind[c] + offset]
      if violin:
        rects = ax.violinplot(val, vert=1, positions=position, widths=width * 0.85, showmedians=True)
        rects["bodies"][0].set_facecolor(color)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
          vp = rects[partname]
          vp.set_edgecolor(color)
      else:
        #rects = ax.boxplot(val, notch=0, sym='k+', vert=1, whis=1.5, positions=position, widths=width * 0.85, patch_artist=True, flierprops=dict(marker='k+', markersize=markersize, markerfacecolor='k', markeredgecolor='k'), boxprops=dict(facecolor=color), medianprops=dict(color='#000000'))
        #rects = ax.boxplot(val, notch=0, sym='k+', whis=1.5, positions=position, widths=width * 0.85, patch_artist=True, flierprops=dict(marker='k+', markersize=markersize, markeredgecolor='k', markeredgewidth=0.2), boxprops=dict(facecolor=color, linewidth=linewidth), capprops=dict(linewidth=linewidth), medianprops=dict(color='#000000', linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), vert=not horizontal)
        #logging.info('plotting %s', val)
        rects = ax.boxplot(val, notch=0, sym='k+', whis=1.5, positions=position, widths=width * 0.85, patch_artist=True, flierprops=dict(marker='k+', markersize=markersize, markeredgecolor='k', markeredgewidth=0.2), boxprops=dict(facecolor=color, linewidth=linewidth), capprops=dict(linewidth=linewidth), medianprops=dict(color='#000000', linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), vert=not horizontal)
      positions.extend(position)
      boxes.append(rects)
    #for rect in rects:
    #  height = rect.get_height()
    #  ax.annotate('{}'.format(height),
    #    xy=(rect.get_x() + rect.get_width() / 2, height),
    #    xytext=(0, 3),  # use 3 points offset
    #    textcoords="offset points",  # in both directions
    #    ha='center', va='bottom')

  # add significance if included
  if significance is not None:
    for sig in significance:
      col1, col2, text = sig.split(',', 2)
      if ',' in text:
        text, custom_y = text.split(',')
      else:
        custom_y = None
      x1, x2 = positions[int(col1)] + 0.1, positions[int(col2)] - 0.1
      if sig_ends is None:
        sig_ends = 0.01
      squiggem = max_zval * sig_ends
      y, h, col = max_zval + squiggem, squiggem, 'k' # TODO these should be scaled to y axis size
      if custom_y is not None:
        y = float(custom_y)
      if significance_nobar:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=linewidth, c=col, alpha=0)
      else:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=linewidth, c=col)
      plt.text((x1+x2)*.5, y+h, text.replace('/', '\n'), ha='center', va='bottom', color=col, fontdict={'size': sig_fontsize})

  if annotate is not None:
    # currently ignore annotate_location
    plt.text(width * 0.5, 0, annotate, ha='center', va='bottom', color='k', fontdict={'size': fontsize})

  # Add some text for labels, title and custom x-axis tick labels, etc.
  if y_label is not None:
    ax.set_ylabel(y_label, fontsize=fontsize)
  if x_label is not None:
    ax.set_xlabel(x_label, fontsize=fontsize)
  ax.set_title(title, fontsize=fontsize)
  ax.set_xticks(ind)
  ax.set_xticklabels([x.replace('/', '\n') for x in xvals], rotation=x_label_rotation) # can use / for eol
  ax.set_xlim((-1, max(ind) + 1 + width))

  # must do this after plotting
  to_include = []
  if include_zero:
    to_include.append(0)
  if include_other is not None:
    to_include.append(include_other)

  if len(to_include) > 0:
    if max_zval < max(to_include):
      ax.set_ylim(ymax=max(to_include))

    if min_zval > min(to_include):
      ax.set_ylim(ymin=min(to_include))

  if y_max is not None:
    ax.set_ylim(ymax=y_max)

  # place legend at right based on https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box/10154763#10154763
  #handles, labels = ax.get_legend_handles_labels()
  #handles = [boxes[0]["boxes"][0], boxes[2]["boxes"][0]]
  if violin:
    handles = [boxes[c]["bodies"][0] for c in range(0, len(boxes), len(xvals))]
  else:
    handles = [boxes[c]["boxes"][0] for c in range(0, len(boxes), len(xvals))]

  if y_counts:
    labels = ['{} ({})'.format(y, int(yvals_count[y] / len(xvals))) for y in yvals]
  else:
    labels = yvals
  if not no_legend:
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01,1.0), borderaxespad=0)
    lgd.get_frame().set_edgecolor('#000000')

  #fig = plt.figure(figsize=(figsize, 1 + int(figsize * len(yvals) / len(xvals))))
  #ax = fig.add_subplot(111)

  logging.info('done processing %i of %i. plot at dpi %i', included, total, dpi)
  plt.tight_layout()
  plt.savefig(target, dpi=dpi)
  matplotlib.pyplot.close('all')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot a bar chart')
  parser.add_argument('--x', required=True, help='category column name')
  parser.add_argument('--y', required=False, help='subcategory column name')
  parser.add_argument('--z', required=True, help='value column name')
  parser.add_argument('--title', required=False, help='graph title')
  parser.add_argument('--annotate', required=False, help='add text to graph')
  parser.add_argument('--annotate_location', required=False, help='text location x,y')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--x_order', required=False, nargs='*', help='order of x axis')
  parser.add_argument('--y_order', required=False, nargs='*', help='order of y axis')
  parser.add_argument('--separator', action='store_true', help='vertical separator')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--height', required=False, type=float, default=8, help='height of plot')
  parser.add_argument('--width', required=False, type=float, default=12, help='width of plot')
  parser.add_argument('--fontsize', required=False, type=int, default=8, help='font size')
  parser.add_argument('--sig_fontsize', required=False, type=int, default=8, help='font size')
  parser.add_argument('--include_zero', action='store_true', help='require zero on y-axis')
  parser.add_argument('--include_other', type=float, required=False, help='require some other value on y-axis')
  parser.add_argument('--significance', required=False, nargs='*', help='add significance of the form col1,col2,text... column numbering follows all leftmost columns from each group, then the next leftmost, finishes with all rightmost')
  parser.add_argument('--significance_nobar', action='store_true', help='do not include bars')
  parser.add_argument('--x_label_rotation', required=False, default='vertical', help='rotation of x labels vertical or horizontal')
  parser.add_argument('--y_log', action='store_true', help='log y scale')
  parser.add_argument('--y_max', required=False, type=float, help='max for y')
  parser.add_argument('--y_counts', action='store_true', help='include counts in legend')
  parser.add_argument('--violin', action='store_true', help='plot as violin plot')
  parser.add_argument('--color_index', default=0, type=int, help='color to start with')
  parser.add_argument('--colors', default=COLORS, nargs='+', help='colors')
  parser.add_argument('--colors_special', default=None, nargs='+', help='define all colors')
  parser.add_argument('--significance_ends', type=float, help='length of sig ends')
  parser.add_argument('--markersize', type=int, default=6, help='size of outliers')
  parser.add_argument('--linewidth', default=1, type=float, help='line width')
  parser.add_argument('--dpi', default=300, type=float, help='dpi')
  parser.add_argument('--horizontal', action='store_true', help='horizontal boxplot') # doesn't work
  parser.add_argument('--no_legend', action='store_true', help='no legend')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  plot_box(sys.stdin, args.target, args.x, args.y, args.z, args.title, args.x_label, args.y_label, args.x_order, args.y_order, args.width, args.height, args.fontsize, args.significance, args.significance_nobar, args.separator, args.include_zero, args.x_label_rotation, args.y_log, args.annotate, args.annotate_location, args.include_other, args.violin, args.y_counts, args.color_index, args.colors, args.y_max, args.significance_ends, args.colors_special, args.no_legend, args.sig_fontsize, args.markersize, args.linewidth, args.dpi, args.horizontal)

