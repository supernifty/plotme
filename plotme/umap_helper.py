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

def main(ifh, ofh, cols, prefix, cluster, exclude, normalise):
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
  embedding = umap.UMAP(random_state=42).fit_transform(raw)
  logging.info('umap calculation: done')
  
  new_cols = ['{}umap0'.format(prefix), '{}umap1'.format(prefix)]
  if cluster:
    logging.info('cluster calculation...')
    hdb = sklearn.cluster.HDBSCAN() # min_cluster_size? 
    hdb.fit(raw)
    new_cols.append('{}cluster'.format(prefix))
    new_cols.append('{}cluster_prob'.format(prefix))
    logging.info('cluster calculation: done')
    zipped = zip(rows, embedding, hdb.labels_, hdb.probabilities_)
  else:
    zipped = zip(rows, embedding)

  odw = csv.DictWriter(ofh, delimiter='\t', fieldnames=idr.fieldnames + new_cols)
  odw.writeheader()
  for ru in zipped:
    ru[0]['{}umap0'.format(prefix)] = ru[1][0]
    ru[0]['{}umap1'.format(prefix)] = ru[1][1]
    if cluster:
      ru[0]['{}cluster'.format(prefix)] = ru[2]
      ru[0]['{}cluster_prob'.format(prefix)] = ru[3]
    odw.writerow(ru[0])

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='generate umap embedding from provided columns')
  parser.add_argument('--cols', required=False, nargs='+', help='which columns to include')
  parser.add_argument('--exclude', action='store_true', help='exclude specified cols')
  parser.add_argument('--prefix', required=False, default='', help='prepend to start of added column names')
  parser.add_argument('--cluster', action='store_true', help='also cluster the data')
  parser.add_argument('--normalise', action='store_true', help='normalise features')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, sys.stdout, args.cols, args.prefix, args.cluster, args.exclude, args.normalise)

