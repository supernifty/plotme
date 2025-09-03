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

def main(ifh, target, x, y, z, color, title, label, color_map=None, markersize=None):
  logging.info('starting...')
  idr = csv.DictReader(ifh, delimiter='\t')
  data = {'x': [], 'y': [], 'z': [], 'color': [], 'size': [], 'symbol': [], 'label': [], 'size': []}
  for r in idr:
    data['x'].append(float(r[x]))
    data['y'].append(float(r[y]))
    data['z'].append(float(r[z]))
    data['label'].append(r[label]) # hover
    data['color'].append(r[color]) # category lookup
    data['symbol'].append(r[color]) # symbol lookup
    data['size'].append(markersize or 4) 

  if color_map is not None:
    color_discrete_map={}
    symbol_map={}
    for m in color_map:
      cat, cs = m.split('=')
      catcol, catsym = cs.split('/')
      color_discrete_map[cat] = catcol
      symbol_map[cat] = catsym
  else:
    color_discrete_map=None
    symbol_map=None

  logging.info('plotting... %s', symbol_map)
  fig = px.scatter_3d(
    x=data['x'], 
    y=data['y'], 
    z=data['z'], 
    color=data['color'], 
    symbol=data['symbol'],
    hover_name=data['label'], 
    size=data['size'],
    title=title, 
    labels={'x': x, 'y': y, 'z': z}, 
    opacity=0.7,
    color_discrete_map=color_discrete_map,
    symbol_map=symbol_map,
  )
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
  parser.add_argument('--color', help='category column for color')
  parser.add_argument('--color_map', required=False, nargs='+', help='specify colors for categories of the form category=color/symbol')
  parser.add_argument('--markersize', type=float, required=False, help='marker size')
  parser.add_argument('--target', help='output file')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(sys.stdin, args.target, args.x, args.y, args.z, args.color, args.title, args.label, args.color_map, args.markersize)
