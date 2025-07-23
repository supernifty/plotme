#!/usr/bin/env python
'''
  interactive 3d graph using plotly
'''

import argparse
import collections
import csv
import logging
import sys

import plotly.express as px

def main(ifh, target, x, y, z, color, title, label):
  logging.info('starting...')
  idr = csv.DictReader(ifh, delimiter='\t')
  data = {'x': [], 'y': [], 'z': [], 'color': [], 'size': [], 'symbol': [], 'label': []}
  for r in idr:
    data['x'].append(float(r[x]))
    data['y'].append(float(r[y]))
    data['z'].append(float(r[z]))
    data['label'].append(r[label])
    data['color'].append(r[color])
    data['symbol'].append(r[color])

  logging.info('plotting...')
  fig = px.scatter_3d(x=data['x'], y=data['y'], z=data['z'], color=data['color'], hover_name=data['label'], title=title, labels={'x': x, 'y': y, 'z': z}, opacity=0.7)
  #fig.update_layout(margin=dict(l=0.1, r=0.1, b=0.1, t=0.1))
  fig.update_layout(
    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)  # default is around 1.25–1.5
    )
  )
  fig.write_html(target) #, auto_open=True)

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='plotly based 3d scatter')
  parser.add_argument('--x', help='col')
  parser.add_argument('--y', help='col')
  parser.add_argument('--z', help='col')
  parser.add_argument('--label', help='col')
  parser.add_argument('--title', help='col')
  parser.add_argument('--color', help='col')
  parser.add_argument('--target', help='output file')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.target, args.x, args.y, args.z, args.color, args.title, args.label)
