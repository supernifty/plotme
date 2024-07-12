#!/usr/bin/env python

import argparse
import collections
import csv
import logging
import sys

import matplotlib.pyplot as plt

def main(ifh, col, target, title, order, colors, nolegend):
  # CLNSIG  Count   Pct
  # Pathogenic      133     0.055
  
  rs = collections.defaultdict(float)
  logging.info('reading stdin...')
  for r in csv.DictReader(ifh, delimiter='\t'):
    
    rs[r[col]] += 1
  
  logging.info('generating %s from %s...', target, rs)
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(111) 
  if order is None:
    labels = sorted([k for k in rs])
  else:
    labels = [k for k in order if k in rs]
    if colors is not None:
      colors = [k[1] for k in zip(order, colors) if k[0] in rs]
  values = [rs[k] for k in labels]

  ax.pie(values, labels=labels, colors=colors, autopct='%.0f%%', labeldistance=None, textprops={'fontsize': 16})
  
  #ax.pie([rs[x] for x in ORDER], labels=ORDER, colors=[COLOR[v] for v in ORDER], autopct='%.0f%%', labeldistance=None, textprops={'fontsize': 16})
  #ax.legend(labels=ORDER, loc='center right')
  if not args.nolegend:
    ax.legend(labels=labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=14)
  if title is not None:
    plt.title(title)
  #plt.tight_layout()
  plt.savefig(target, bbox_inches="tight")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='pie graph')
  parser.add_argument('--target', required=True, help='more logging')
  parser.add_argument('--col', required=True, help='more logging')
  parser.add_argument('--title', required=False, help='more logging')
  parser.add_argument('--order', required=False, nargs='*', help='more logging')
  parser.add_argument('--colors', required=False, nargs='*', help='more logging')
  parser.add_argument('--nolegend', action='store_true', help='more logging')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.col, args.target, args.title, args.order, args.colors, args.nolegend)
