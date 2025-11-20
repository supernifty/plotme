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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
import umap

from statsmodels.stats.multitest import multipletests

import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing
from sklearn.metrics import silhouette_samples
from sklearn.utils import resample

import hdbscan

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


##### measuring a p-value for each cluster #####

def cluster_stat(X, labels, min_cluster_size=10, stat="silhouette"):
    """Compute cluster-level statistic (mean silhouette or HDBSCAN stability)."""
    stats = {}
    if stat == "silhouette":
        sil = silhouette_samples(X, labels)
        #for c in np.unique(labels[labels >= 0]):
        for c in np.unique(labels):
            stats[c] = np.mean(sil[labels == c])
    elif stat == "stability":
        # HDBSCAN exposes cluster persistence as stability
        # but we need to map it to cluster labels
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size).fit(X)
        for c, s in zip(clusterer.labels_, clusterer.probabilities_):
            #if c >= 0:
            stats.setdefault(c, []).append(s)
        stats = {c: np.mean(vals) for c, vals in stats.items()}
    else:
        raise ValueError("stat must be 'silhouette' or 'stability'")
    return stats


def cluster_significance_test(X, min_cluster_size=10, n_null=500, stat="silhouette", null_model="bootstrap", fdr=True, random_state=None):
    """
    Test statistical significance of HDBSCAN clusters via null resampling.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    min_cluster_size : int
        HDBSCAN minimum cluster size.
    n_null : int
        Number of null replicates.
    stat : str
        Statistic: 'silhouette' or 'stability'.
    null_model : str
        Null generation method: 'bootstrap' or 'gaussian'.
    fdr : bool
        Apply FDR correction to p-values.
    random_state : int or None
        Random seed.

    Returns
    -------
    dict
        Cluster IDs mapped to (statistic, raw p-value, corrected p-value if fdr).
    """
    rng = np.random.default_rng(random_state)
    # observed clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size).fit(X) # to match main()

    counts = collections.defaultdict(int)
    for l in clusterer.labels_:
      counts[l] += 1
    min_size = min([x for x in counts if x != -1])
    max_size = max([x for x in counts if x != -1])
    logging.info('significance cluster umap calculation: done. %i clusters with %i unassigned. sizes: %s', len(counts) - 1, counts[-1], counts)

    obs_stats = cluster_stat(X, clusterer.labels_, min_cluster_size=min_cluster_size, stat=stat)
    logging.info('obs_stats is %s', obs_stats)

    # build null distribution stratified by cluster size
    null_stats_by_size = collections.defaultdict(list)

    for i in range(n_null):
        if null_model == "bootstrap":
            X_null = resample(X, replace=True, random_state=rng.integers(1e9))
        elif null_model == "gaussian":
            X_null = rng.multivariate_normal(X.mean(axis=0),
                                             np.cov(X.T),
                                             size=X.shape[0])
        else:
            raise ValueError("null_model must be 'bootstrap' or 'gaussian'")

        clusterer_null = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size).fit(X_null)
        null_stat = cluster_stat(X_null, clusterer_null.labels_, stat=stat)

        for c, val in null_stat.items():
            size = np.sum(clusterer_null.labels_ == c)
            null_stats_by_size[size].append(val)

    # compute raw p-values
    raw_pvals = {}
    for c, val in obs_stats.items():
        size = np.sum(clusterer.labels_ == c)
        # match null cluster size approximately
        nearest_size = min(null_stats_by_size.keys(), key=lambda s: abs(s - size))
        null_vals = np.array(null_stats_by_size[nearest_size])
        raw_pvals[c] = np.mean(null_vals >= val)

    # multiple testing correction
    if fdr and len(raw_pvals) > 1:
        clusters, pvals = zip(*raw_pvals.items())
        _, corrected, _, _ = multipletests(pvals, method="fdr_bh")
        results = {c: (obs_stats[c], raw_pvals[c], corr)
                   for c, corr in zip(clusters, corrected)}
    else:
        results = {c: (obs_stats[c], raw_pvals[c], None)
                   for c in raw_pvals}

    return results


##########

def safe_fit(reducer, raw): 
  with open(os.devnull, "w") as fnull:
    with redirect_stdout(fnull):
      reducer.fit(raw)

def main(ifh, ofh, cols, prefix, cluster, exclude, normalise, cluster_umap, min_cluster_size, dim=2, algorithm='umap', cluster_settings=False, test_file=None, cluster_settings_dim=2, cluster_settings_n_neighbours=30, cluster_significance=False):
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
    #reducer = umap.UMAP(random_state=SEED, n_components=dim)
    reducer = umap.ParametricUMAP(random_state=SEED, n_components=dim)
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
      clusterable_reducer = umap.ParametricUMAP(
      #clusterable_reducer = umap.UMAP(
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

    # pvalues!
    if cluster_significance is not None:
      logging.info('calculating significance...')
      #sig = cluster_significance_test(clusterable_embedding, min_cluster_size=min_cluster_size, n_null=500, stat="stability", null_model="bootstrap", fdr=True, random_state=SEED)
      stability = cluster_stat(clusterable_embedding, hdb.labels_, min_cluster_size=min_cluster_size, stat="stability")
      silhouette = cluster_stat(clusterable_embedding, hdb.labels_, min_cluster_size=min_cluster_size, stat="silhouette")
      odw = csv.DictWriter(open(cluster_significance, 'wt'), delimiter='\t', fieldnames=['cluster', 'silhouette', 'stability'])
      odw.writeheader()
      for i in sorted(np.unique(hdb.labels_)):
        odw.writerow({'cluster': i, 'silhouette': silhouette.get(i, 'n/a'), 'stability': stability.get(i, 'n/a')})
      logging.info('calculating significance: done')

    # summarise
    counts = collections.defaultdict(int)
    sizes = (9999, 0)
    for l in hdb.labels_:
      counts[l] += 1
    min_size = min([x for x in counts if x != -1])
    max_size = max([x for x in counts if x != -1])
    logging.info('cluster umap calculation: done. %i clusters with %i unassigned. sizes: %s', len(counts) - 1, counts[-1], counts)
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
      n = 2
      if cluster:
        ru[0]['{}cluster'.format(prefix)] = ru[2]
        ru[0]['{}cluster_prob'.format(prefix)] = ru[3]
        n += 2
      if cluster_umap: # umap clusterable will end up here if cluster_settings are enabled
        ru[0]['{}cluster_umap'.format(prefix)] = ru[n]
        ru[0]['{}cluster_umap_prob'.format(prefix)] = ru[n + 1]
        if cluster_settings:
          for cn in range(cluster_settings_dim):
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
        r['{}cluster_umap'.format(prefix)] = 'unassigned' if labels[i] == '-1' else labels[i]
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
  parser.add_argument('--cluster_significance', help='calculate p-value with bootstrapping and write to filename')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, sys.stdout, args.cols, args.prefix, args.cluster, args.exclude, args.normalise, args.cluster_umap, args.minimum_cluster_size, args.dim, args.algorithm, args.cluster_settings, args.test_file, args.cluster_settings_dim, args.cluster_settings_n_neighbours, args.cluster_significance)

