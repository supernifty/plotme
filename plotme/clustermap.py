#!/usr/bin/env python

import argparse
import collections
import logging
import sys

import scipy.cluster.hierarchy
import scipy.spatial.distance
import sklearn.metrics

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def choose_k(data, col_linkage, min_cluster_size=4):
  # distance matrix between columns
  X = data.T.values  # cluster columns
  distance_matrix = scipy.spatial.distance.pdist(X, metric="cosine")
  scores = []

  best = (None, None)
  for k in range(2, 50):
    labels = scipy.cluster.hierarchy.fcluster(col_linkage, k, criterion="maxclust")
    counts = collections.Counter(labels)
    min_size = min(counts.values())
    #logging.info('k=%i min=%i', k, min_size)
    if min_size < min_cluster_size:
      continue # or could break?
    score = sklearn.metrics.silhouette_score(X, labels, metric="cosine")
    scores.append(score)
    if best[0] is None or score > best[0]:
      best = (score, k)

  if best[1] is None:
    logging.warning('no cluster possible with min_cluster_size %i', min_cluster_size)
    return choose_k(data, col_linkage, min_cluster_size=min_cluster_size-1)

  logging.info('best is %s from %s', best, scores)

  return best[1]

def plot_clustermap(ifh, delimiter, target, y, indicator, width, height, switch, separate=True, min_cluster_size=4, write_cluster=None):
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
  #x = sns.clustermap(df, standard_scale=1, cmap='plasma') # normalise
  #x = sns.clustermap(df, z_score=1, cmap='vlag', metric="cosine", row_colors=row_colors, figsize=(width, height)) # normalise
  #x = sns.clustermap(df, z_score=1, cmap='vlag', method='average', metric="correlation", row_colors=row_colors, figsize=(width, height)) # normalise
  if switch:
    df = df.T
  x = sns.clustermap(df, z_score=1, cmap='plasma', metric="cosine", dendrogram_ratio=(.1, .1), row_colors=row_colors, figsize=(width, height)) # normalise

  if separate:
    col_linkage = x.dendrogram_col.linkage
    k = choose_k(df, col_linkage, min_cluster_size)
    col_clusters = scipy.cluster.hierarchy.fcluster(col_linkage, k, criterion="maxclust")
    if write_cluster is not None:
      cluster_series = pd.Series(col_clusters, index=df.columns, name="cluster")
      cluster_series.to_csv(write_cluster, sep=delimiter, index=True)
    ordered_cols = col_clusters[x.dendrogram_col.reordered_ind]
    boundaries = np.where(np.diff(ordered_cols))[0]
    for b in boundaries:
      x.ax_heatmap.vlines(b + 1, *x.ax_heatmap.get_ylim(), colors='black', linewidth=2)

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
  parser.add_argument('--separate', action='store_true', help='separate clusters')
  parser.add_argument('--min_cluster_size', default=4, type=int, help='min cluster size')
  parser.add_argument('--write_cluster', required=False, help='write cluster assignments to filename')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
  plot_clustermap(sys.stdin, args.delimiter, args.target, args.y, args.indicator, args.width, args.height, args.switch_axes, args.separate, args.min_cluster_size, args.write_cluster)
