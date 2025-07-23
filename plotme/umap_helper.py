#!/usr/bin/env python
'''
  apply umap dimensionality reduction
'''

import argparse
import csv
import logging
import sys

import umap

import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing

def main(ifh, ofh, cols, prefix, cluster, exclude, normalise, cluster_umap, min_cluster_size, dim=2):
  logging.info('starting...')
  raw = []
  rows = []
  idr = csv.DictReader(ifh, delimiter='\t')
  for r in idr:
    rows.append(r)
    if exclude:
      raw.append([float(r[x]) for x in r if x not in cols])
    else:
      raw.append([float(r[x]) for x in cols])

  # normalise!
  if normalise:
    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(raw)
    raw = scaled

  logging.info('umap calculation...')
  embedding = umap.UMAP(random_state=42, n_components=dim).fit_transform(raw)
  logging.info('umap calculation: done')
  
  new_cols = []
  for n in range(dim):
    new_cols.append('{}umap{}'.format(prefix, n))

  if cluster:
    logging.info('cluster calculation...')
    cluster_raw = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size) # min_cluster_size? 
    cluster_raw.fit(raw) # original data
    new_cols.append('{}cluster'.format(prefix))
    new_cols.append('{}cluster_prob'.format(prefix))
    logging.info('cluster calculation: done')
    zipped = zip(rows, embedding, cluster_raw.labels_, cluster_raw.probabilities_)
  else:
    zipped = zip(rows, embedding)

  if cluster_umap: # use the umap custering
    logging.info('cluster umap calculation from %i embeddings...', len(embedding))
    hdb = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size) # min_cluster_size? 
    hdb.fit(embedding) # umap fit
    new_cols.append('{}cluster_umap'.format(prefix))
    new_cols.append('{}cluster_umap_prob'.format(prefix))
    logging.info('cluster umap calculation: done')
    if cluster:
      zipped = zip(rows, embedding, cluster_raw.labels_, cluster_raw.probabilities_, hdb.labels_, hdb.probabilities_)
    else:
      zipped = zip(rows, embedding, hdb.labels_, hdb.probabilities_)

  odw = csv.DictWriter(ofh, delimiter='\t', fieldnames=idr.fieldnames + new_cols)
  odw.writeheader()
  for ru in zipped: # add additional findings to ru[0] - this is such a messy way to do this :-/
    # embeddings
    for n in range(dim):
      ru[0]['{}umap{}'.format(prefix, n)] = ru[1][n]
    n = dim
    if cluster:
      ru[0]['{}cluster'.format(prefix)] = ru[2]
      ru[0]['{}cluster_prob'.format(prefix)] = ru[3]
      n += 2
    if cluster_umap:
      ru[0]['{}cluster_umap'.format(prefix)] = ru[n]
      ru[0]['{}cluster_umap_prob'.format(prefix)] = ru[n + 1]
    odw.writerow(ru[0])

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='generate umap embedding from provided columns')
  parser.add_argument('--cols', required=False, nargs='+', help='which columns to include')
  parser.add_argument('--exclude', action='store_true', help='exclude specified cols')
  parser.add_argument('--prefix', required=False, default='', help='prepend to start of added column names')
  parser.add_argument('--cluster', action='store_true', help='also cluster the data')
  parser.add_argument('--cluster_umap', action='store_true', help='cluster the umap results')
  parser.add_argument('--minimum_cluster_size', type=int, default=5, required=False, help='minimum membership of any cluster')
  parser.add_argument('--normalise', action='store_true', help='normalise features')
  parser.add_argument('--dim', required=False, default=2, type=int, help='umap dimensions')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, sys.stdout, args.cols, args.prefix, args.cluster, args.exclude, args.normalise, args.cluster_umap, args.minimum_cluster_size, args.dim)

