#!/usr/bin/env python
'''
  apply umap dimensionality reduction
'''

import argparse
import collections
import csv
import logging
import os
import random
import sys

from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
import umap

import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing

import hdbscan

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def safe_fit(reducer, raw): 
  with open(os.devnull, "w") as fnull:
    with redirect_stdout(fnull):
      reducer.fit(raw)

def main(ifh, ofh, cols, prefix, cluster, exclude, normalise, cluster_umap, min_cluster_size, dim=2, algorithm='umap', cluster_settings=False, test_file=None, cluster_settings_dim=2, cluster_settings_n_neighbours=30):
  logging.info('starting...')
  raw = []
  rows = []
  idr = csv.DictReader(ifh, delimiter='\t')
  for r in idr:
    rows.append(r)
    if exclude:
      raw.append([float(r[x]) for x in r if x not in cols])
    else:
      raw.append([float(r[x]) for x in cols if x in r]) # silently ignores missing cols

  logging.info('considering %i cols', len(raw[0]))

  # normalise!
  if normalise:
    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(raw)
    raw = scaled

  # dimensionality reduction
  logging.info('dimensionality reduction on %i rows...', len(raw))
  if algorithm == 'umap':
    reducer = umap.UMAP(random_state=SEED, n_components=dim)
    #reducer = umap.ParametricUMAP(random_state=SEED, n_components=dim)
  elif algorithm == 'pca':
    reducer = sklearn.decomposition.PCA(n_components=dim)
  else:
    logging.fatal('unsupported algorithm %s', algorithm)

  safe_fit(reducer, raw)
  logging.info('row0: %s', raw[0])
  embedding = reducer.transform(raw)
  logging.info('dimensionality reduction: %s', embedding[0])
  logging.info('dimensionality reduction: done')
  
  # clustering
  new_cols = []
  for n in range(dim):
    new_cols.append('{}umap{}'.format(prefix, n))

  if cluster:
    logging.info('cluster calculation...')
    #cluster_raw = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=test_file is not None) # min_cluster_size? 
    cluster_raw = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=test_file is not None) # min_cluster_size? 
    cluster_raw.fit(raw) # original data
    new_cols.append('{}cluster'.format(prefix))
    new_cols.append('{}cluster_prob'.format(prefix))
    logging.info('cluster calculation: done')
    zipped = zip(rows, embedding, cluster_raw.labels_, cluster_raw.probabilities_)
  else:
    zipped = zip(rows, embedding)

  if cluster_umap: # use the umap custering
    logging.info('cluster umap calculation from %i embeddings...', len(embedding))
    
    # use clusterable embedding as recommended by https://umap-learn.readthedocs.io/en/latest/clustering.html
    if cluster_settings:
      #clusterable_reducer = umap.ParametricUMAP(
      clusterable_reducer = umap.UMAP(
        n_neighbors=cluster_settings_n_neighbours,
        min_dist=0.0,
        n_components=cluster_settings_dim,
        random_state=SEED,
      )
      safe_fit(clusterable_reducer, raw)
      clusterable_embedding = clusterable_reducer.transform(raw)
      for n in range(dim):
        new_cols.append('{}umap_clusterable{}'.format(prefix, n))
    else:
      clusterable_embedding = embedding
 
    #hdb = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size) # min_cluster_size? 
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size, prediction_data=test_file is not None) # min_cluster_size? 
    hdb.fit(clusterable_embedding) # umap fit
    new_cols.append('{}cluster_umap'.format(prefix))
    new_cols.append('{}cluster_umap_prob'.format(prefix))

    # summarise
    counts = collections.defaultdict(int)
    sizes = (9999, 0)
    for l in hdb.labels_:
      counts[l] += 1
    min_size = min([x for x in counts if x != -1])
    max_size = max([x for x in counts if x != -1])
    logging.info('cluster umap calculation: done. %i clusters with %i unassigned. sizes range from %i to %i', len(counts) - 1, counts[-1], min_size, max_size)
    if cluster:
      zipped = zip(rows, embedding, cluster_raw.labels_, cluster_raw.probabilities_, hdb.labels_, hdb.probabilities_)
    else:
      zipped = zip(rows, embedding, hdb.labels_, hdb.probabilities_)

  # write the results
  odw = csv.DictWriter(ofh, delimiter='\t', fieldnames=idr.fieldnames + new_cols)
  odw.writeheader()
  if test_file is None:
    #logging.info(clusterable_embedding)
    for i, ru in enumerate(zipped): # add additional findings to ru[0] - this is such a messy way to do this :-/
      # embeddings
      for n in range(dim):
        ru[0]['{}umap{}'.format(prefix, n)] = ru[1][n]
      n = dim
      if cluster:
        ru[0]['{}cluster'.format(prefix)] = ru[2]
        ru[0]['{}cluster_prob'.format(prefix)] = ru[3]
        n += 2
      if cluster_umap: # umap clusterable will end up here if cluster_settings are enabled
        ru[0]['{}cluster_umap'.format(prefix)] = ru[n]
        ru[0]['{}cluster_umap_prob'.format(prefix)] = ru[n + 1]
        if cluster_settings:
          for cn in range(dim):
            #logging.info('%i %i %s', i, cn, clusterable_embedding[i][cn])
            ru[0]['{}umap_clusterable{}'.format(prefix, cn)] = clusterable_embedding[i][cn]
      odw.writerow(ru[0])
  else:
    logging.info('applying to test data...')
    i = 0
    rows = []
    data = []
    unassigned = 0
    for i, r in enumerate(csv.DictReader(open(test_file, 'rt'), delimiter='\t')):
      if exclude:
        row = [float(r[x]) for x in r if x not in cols]
      else:
        row = [float(r[x]) for x in cols]
      data.append(r)
      rows.append(row)

    # do everything
    if normalise:
      rows = scaler.transform(rows)
      # logging.info('test scaled: %s', rows[0])

    # reduce
    logging.info('row0: %s', rows[0])
    test_embedding = reducer.transform(rows)
    logging.info('test embedding: %s', test_embedding[0])
    #sys.exit(0)

    for i, r in enumerate(data):
      for n in range(dim):
        r['{}umap{}'.format(prefix, n)] = test_embedding[i][n]

    # cluster with prediction
    if cluster:
      labels, strengths = hdbscan.approximate_predict(cluster_raw, rows)

      for i, r in enumerate(data):
        r['{}cluster'.format(prefix)] = labels[i]
        r['{}cluster_prob'.format(prefix)] = strengths[i]

    if cluster_umap:
      if cluster_settings:
        test_embedding = clusterable_reducer.transform(rows)
      else:
        test_embedding = reducer.transform(rows)
      labels, strengths = hdbscan.approximate_predict(hdb, test_embedding)
      #logging.info('%s -> %s %s', test_embedding, labels, strengths)
      for i, r in enumerate(data):
        r['{}cluster_umap'.format(prefix)] = labels[i]
        r['{}cluster_umap_prob'.format(prefix)] = strengths[i]
        if labels[i] == '-1':
         unassigned += 1

    logging.info('writing...')
    for r in data:
      odw.writerow(r)
    logging.info('wrote %i rows with %i unassigned', len(data), unassigned)
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='generate umap embedding from provided columns')
  parser.add_argument('--cols', required=False, nargs='+', help='which columns to include')
  parser.add_argument('--exclude', action='store_true', help='exclude specified cols')
  parser.add_argument('--prefix', required=False, default='', help='prepend to start of added column names')
  parser.add_argument('--cluster', action='store_true', help='also cluster the data')
  parser.add_argument('--cluster_umap', action='store_true', help='cluster the umap results')
  parser.add_argument('--cluster_settings', action='store_true', help='cluster the umap results with specific settings')
  parser.add_argument('--cluster_settings_n_neighbours', type=int, required=False, default=30, help='cluster the umap results with specific settings')
  parser.add_argument('--cluster_settings_dim', required=False, default=2, type=int, help='umap dimensions with clusterer')
  parser.add_argument('--minimum_cluster_size', type=int, default=5, required=False, help='minimum membership of any cluster')
  parser.add_argument('--normalise', action='store_true', help='normalise features')
  parser.add_argument('--dim', required=False, default=2, type=int, help='umap dimensions')
  parser.add_argument('--algorithm', required=False, default='umap', help='set to pca to replace umap')
  parser.add_argument('--test_file', required=False, help='test umap and clustering on this dataset')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, sys.stdout, args.cols, args.prefix, args.cluster, args.exclude, args.normalise, args.cluster_umap, args.minimum_cluster_size, args.dim, args.algorithm, args.cluster_settings, args.test_file, args.cluster_settings_dim, args.cluster_settings_n_neighbours)

