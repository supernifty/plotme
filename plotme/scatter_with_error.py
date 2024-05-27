#!/usr/bin/env python
'''
  given tumour and normal vcf pairs, explore msi status
'''

import argparse
import csv
import logging
import sys

import matplotlib.pyplot as plt
import matplotlib.style

import scipy.stats

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def main(ifh, x, y, yl, yh, hue, xlabel, ylabel, log, title, show_correlation, diagonal, out, markersize, width, height):
  logging.info('starting...')
  matplotlib.style.use('seaborn')
  xs = []
  ys = []
  yls = []
  yhs = []
  hues = []

  for r in csv.DictReader(ifh, delimiter='\t'):
    xs.append(float(r[x]))
    ys.append(float(r[y]))
    yls.append(max(0, float(r[y]) - float(r[yl])))
    yhs.append(float(r[yh]) - float(r[y]))
    if hue is not None:
      hues.append(r[hue])
    else:
      hues.append('')

  fig = plt.figure(figsize=(width, height))
  ax = fig.add_subplot(111)
  if hues is not None:
    seen = []
    for x, y, yl, yh, hue in zip(xs, ys, yls, yhs, hues):
      if hue not in seen:
        seen.append(hue)
        ax.errorbar(x, y, yerr=([yl], [yh]), fmt='^', color=COLORS[seen.index(hue)], label=hue, markersize=markersize, alpha=0.8)
      else:        
        ax.errorbar(x, y, yerr=([yl], [yh]), fmt='^', color=COLORS[seen.index(hue)], markersize=markersize, alpha=0.8)
    ax.legend()
  else:
    ax.errorbar(xs, ys, yerr=(yls, yhs), fmt='o', markersize=markersize, alpha=0.8)

  if diagonal:
    maxes = max([max(xs), max(ys)])
    ax.plot([0, maxes], [0, maxes], ls="--", c=".6")# , transform=ax.transAxes)

  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if title is not None:
    correlation = scipy.stats.pearsonr(xs, ys)
    ax.set_title('{} (n={} correlation={:.2f})'.format(title, len(xs), correlation[0]))
  if show_correlation:
    correlation = scipy.stats.pearsonr(xs, ys)
    logging.info('correlation is %f', correlation[0])
    ax.annotate('n={} correlation={:.3f}'.format(len(xs), correlation[0]), (0, 0), transform=ax.transAxes)


  if log: # does this work?
    ax.set_yscale('log')
    ax.set_xscale('log')

  plt.savefig(out)

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Scatter with error bars on y')
  parser.add_argument('--x', required=True, help='x col')
  parser.add_argument('--y', required=True, help='y col')
  parser.add_argument('--yl', required=True, help='y lower bound')
  parser.add_argument('--yh', required=True, help='y upper bound')
  parser.add_argument('--hue', required=False, help='colour col')
  parser.add_argument('--out', required=True, help='output file')
  parser.add_argument('--x_label', required=False, help='x axis label')
  parser.add_argument('--y_label', required=False, help='y axis label')
  parser.add_argument('--title', required=False, help='graph title')
  parser.add_argument('--show_correlation', action='store_true', help='annotate with x and y correlation')
  parser.add_argument('--diagonal', action='store_true', help='draw a diaganal line')
  parser.add_argument('--markersize', required=False, default=20, type=int, help='fontsize')
  parser.add_argument('--width', required=False, default=12, type=float, help='figsize width')
  parser.add_argument('--height', required=False, default=8, type=float, help='figsize width')
  parser.add_argument('--log', action='store_true', help='log scales')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.x, args.y, args.yl, args.yh, args.hue, args.x_label, args.y_label, args.log, args.title, args.show_correlation, args.diagonal, args.out, args.markersize, args.width, args.height)
