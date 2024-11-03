#!/usr/bin/env python
'''
  figure out pairs and plot variables
  
  calculate significance for each variable change?
'''

import argparse
import collections
import csv
import logging
import random
import sys

import matplotlib.pyplot as plt

COLORS = ['#c1009b', '#305ec1', '#909090', '#5e9b5e', '#9b9b00']

X_JITTER=0.1
Y_JITTER=0.2
TEXT_JITTER=0.02

FACTOR=4

def jitter(x, d):
  #return max(0.01, x + d * (random.random() - 0.5))
  return x + d * random.random() # strictly increasing

def main(pairs_fh, x1, x2, target, title, group, x1_name=None, x2_name=None):
  logging.info('starting...')
  #  plot each relationship
  fig = plt.figure()
  ax = fig.add_subplot(111)

  x1_name = x1 if x1_name is None else x1_name
  x2_name = x2 if x2_name is None else x2_name

  relationship = collections.defaultdict(int)
  nodes = collections.defaultdict(int)
  for r in csv.DictReader(pairs_fh, delimiter='\t'):
    relationship[(int(r[x1]), int(r[x2]))] += 1
    nodes['{} {}'.format(x1_name, r[x1])] += 1
    nodes['{} {}'.format(x2_name, r[x2])] += 1
    if not group:
      ax.plot((x1_name, x2_name), (jitter(int(r[x1]), Y_JITTER), jitter(int(r[x2]), Y_JITTER)), color=COLORS[int(r[x1])], marker='s', alpha=0.5, linewidth=4)

  for node in nodes:
    sys.stdout.write('node {}: {}\n'.format(node, nodes[node]))

  if group:
    for r in relationship:
      xs = (x1_name, x2_name)
      ys = r
      ax.plot(xs, ys, color=COLORS[ys[0]], marker='s', alpha=0.5, linewidth=relationship[r] * FACTOR)
      sys.stdout.write('relationship {} -> {}: {}\n'.format(r[0], r[1], relationship[r]))

  #ax.legend()
  if title is not None:
    ax.set_title(title)
  plt.tight_layout()
  plt.savefig(target, dpi=300, transparent=False)
  logging.info('done generating %s', target)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Assess MSI')
  parser.add_argument('--x1', required=True, help='x column')
  parser.add_argument('--x2', required=True, help='x column')
  parser.add_argument('--x1_name', required=False, help='display name')
  parser.add_argument('--x2_name', required=False, help='display name')
  parser.add_argument('--title', required=False)
  parser.add_argument('--target', required=False, default='plot.png', help='target')
  parser.add_argument('--group', action='store_true', help='group results')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.x1, args.x2, args.target, args.title, args.group, args.x1_name, args.x2_name)

