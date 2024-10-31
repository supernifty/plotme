#!/usr/bin/env python
'''
  given tumour and normal vcf pairs, explore msi status
'''

import argparse
import csv
import logging
import sys

import numpy as np
import pandas as pd
import seaborn as sns

def main(value, group, target):
  logging.info('reading stdin...')
  data = []
  for r in csv.DictReader(sys.stdin, delimiter='\t'):
    data.append([float(r[value]), 'all' if group is None else r[group]])

  data = pd.DataFrame(data, index=list(range(len(data))), columns=[value, 'group'])
  
  sns.set_style('whitegrid')
  logging.info('plotting...')
  #plt = sns.kdeplot(data, x=value, hue=group, legend=True)
  #plt = sns.kdeplot(data, x=value, legend=True)
  #plt = sns.kdeplot(data, x='value', hue='group', legend=True)
  plt = sns.kdeplot(x=data[value], hue=data['group'], legend=True, fill=True, common_norm=False, alpha=.5)
  fig = plt.get_figure()
  logging.info('saving...')
  fig.savefig(target) 
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Assess MSI')
  parser.add_argument('--value', required=True, help='tumour vcf')
  parser.add_argument('--group', required=False, help='tumour vcf')
  parser.add_argument('--target', required=False, default='plot.png', help='tumour vcf')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.value, args.group, args.target)

