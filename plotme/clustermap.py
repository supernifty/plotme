#!/usr/bin/env python

import argparse
import logging
import sys

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_clustermap(ifh, delimiter, target, y, indicator, width, height, switch):
  logging.info('reading from stdin...')
  df = pd.read_csv(ifh, sep=delimiter)
  df = df.set_index(y)

  if indicator is not None:
    logging.info('df %s...', df)
    dfi = df[indicator]
    df = df.drop(indicator)
    lut = dict(zip(dfi.unique(), COLORS[:len(dfi.unique())]))
    row_colors = dfi.map(lut)
    sns.clustermap(iris, row_colors=row_colors)
  else:
    row_colors = None

  # standardise
  logging.info('clustering %s...', df)
  #x = sns.clustermap(df, standard_scale=1, cmap='plasma') # normalise
  #x = sns.clustermap(df, z_score=1, cmap='vlag', metric="cosine", row_colors=row_colors, figsize=(width, height)) # normalise
  #x = sns.clustermap(df, z_score=1, cmap='vlag', method='average', metric="correlation", row_colors=row_colors, figsize=(width, height)) # normalise
  if switch:
    df = df.T
  x = sns.clustermap(df, z_score=1, cmap='plasma', metric="cosine", dendrogram_ratio=(.1, .1), row_colors=row_colors, figsize=(width, height)) # normalise
  #x = sns.clustermap(df, cmap='plasma')
  logging.info('saving...')
  x.savefig(target)
  logging.info('done')

# Data set
#url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/mtcars.csv'
#df = pd.read_csv(url)
#df = df.set_index('model')

# Standardize or Normalize every column in the figure
# Standardize:
#x = sns.clustermap(df, standard_scale=1)
#x.savefig('out/clustermap.png')

# Normalize
#sns.clustermap(df, z_score=1)
#plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='clustermap plot')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  parser.add_argument('--indicator', required=False, help='y-axis indicator') # TODO doesn't work
  parser.add_argument('--delimiter', required=False, default='\t', help='input file delimiter')
  parser.add_argument('--width', required=False, default=8, type=int, help='figure width')
  parser.add_argument('--height', required=False, default=8, type=int, help='figure width')
  parser.add_argument('--switch_axes', action='store_true', help='switch_axes')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
  plot_clustermap(sys.stdin, args.delimiter, args.target, args.y, args.indicator, args.width, args.height, args.switch_axes)
