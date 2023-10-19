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

def main(ifh, x, y, yl, yh, out):
  logging.info('starting...')
  matplotlib.style.use('seaborn')
  xs = []
  ys = []
  yls = []
  yhs = []

  for r in csv.DictReader(ifh, delimiter='\t'):
    xs.append(float(r[x]))
    ys.append(float(r[y]))
    yls.append(max(0, float(r[y]) - float(r[yl])))
    yhs.append(float(r[yh]) - float(r[y]))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.errorbar(xs, ys, yerr=(yls, yhs), fmt='o')

  plt.savefig(out)

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Assess MSI')
  parser.add_argument('--x', required=True, help='tumour vcf')
  parser.add_argument('--y', required=True, help='tumour vcf')
  parser.add_argument('--yl', required=True, help='tumour vcf')
  parser.add_argument('--yh', required=True, help='tumour vcf')
  parser.add_argument('--out', required=True, help='tumour vcf')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.x, args.y, args.yl, args.yh, args.out)
