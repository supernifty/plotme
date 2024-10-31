#!/usr/bin/env python
'''
  swimmer plot
'''

import argparse
import collections
import csv
import logging
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams

theme = {
  'ongoing': {'color': 'blue', 'markersize': 64, 'marker': '*'},
  'death': {'color': 'blue', 'markersize': 64, 'marker': 's'},
  'adenoma': {'color': '#882255', 'markersize': 64, 'marker': 'v', 'ydelta': 0.3},
  'colonoscopy': {'color': 'green', 'markersize': 64, 'marker': '+'},
  'CRC': {'color': 'red', 'markersize': 64, 'marker': '^', 'ydelta': -0.3},
  'indicator_case': {'color': '#aa4499', 'markersize': 64, 'marker': 's'},
  'indicator_control': {'color': '#88ccee', 'markersize': 64, 'marker': 's'},
  'indicator_M': {'color': '#2468ce', 'markersize': 64, 'marker': 'v'},
  'indicator_F': {'color': '#ff00ff', 'markersize': 64, 'marker': '^'},
  'indicator_MSH2': {'color': '#00820D', 'markersize': 64, 'marker': 's'},
  'indicator_MLH1': {'color': '#e08000', 'markersize': 64, 'marker': 'o'},
  'indicator_MSH6': {'color': '#8888ee', 'markersize': 64, 'marker': '+'},
}

#sample  event_name      event_value
#P1      sex     male
#P1      adenoma   51
#P1      colonoscopy 51
#P1      CRC     55
#P1      ongoing   80
#P1      sex     female
#P1      adenoma   55
#P1      colonoscopy 56
#P1      CRC     58
#P1      death   60
def main(data, target, width=8, height=6, dpi=72, start=None, indicator_col=None, bar_indicator=None, sort_key=None):
  logging.info('reading %s...', data)
  try:
    matplotlib.style.use('seaborn-v0_8')
  except:
    matplotlib.style.use('seaborn')

  ages = {} # sample end
  events = {} # event sample value
  indicator = {}
  bar_indicators = {}

  if start is None:
    start = 0

  # supported event_names: ongoing death args.indicator adenoma colonoscopy CRC
  for r in csv.DictReader(open(data, 'rt'), delimiter='\t'):
    logging.debug('processing %s', r)
    # make barchart
    if r['event_name'] in ('ongoing', 'death'):
      ages[r['sample']] = float(r['event_value'])
    if r['event_name'] == indicator_col: 
      indicator[r['sample']] = r['event_value']
    if r['event_name'] == bar_indicator: 
      bar_indicators[r['sample']] = r['event_value']
    if r['event_name'] not in ('adenoma', 'colonoscopy', 'CRC'):
      continue
    if r['event_name'] not in events:
      events[r['event_name']] = collections.defaultdict(list)
    #logging.info(r)
    events[r['event_name']][r['sample']].append(float(r['event_value']))

  # todo sorting
  yvals = range(len(ages))
  if sort_key is None:
    labels = sorted(ages.keys(), key=lambda x: ages[x])
  elif sort_key == 'indicator':
    labels = sorted(indicator.keys(), key=lambda x: (indicator[x], ages[x]))[::-1]
  logging.info(labels)

  fig = plt.figure(figsize=(width, height))
  ax = fig.add_subplot(111)

  if bar_indicator is None:
    ax.barh(yvals, [ages[x] - start for x in labels], height=0.1, left=start, align='center', color='grey')
  else:
    ax.barh(yvals, [ages[x] - start for x in labels], height=0.1, left=start, align='center', color=[theme['indicator_{}'.format(bar_indicators[x])]['color'] for x in labels])
  ax.set_yticks(yvals, labels=labels)

  ax.barh([0], [0], height=0.0, left=start, align='center', color=theme['indicator_MLH1']['color'], label='MLH1')
  ax.barh([0], [0], height=0.0, left=start, align='center', color=theme['indicator_MSH2']['color'], label='MSH2')
  

  # events
  for event in events:
    xvals = []
    yvals = []
    for y, sample in enumerate(labels):
      if sample not in events[event]:
        continue
      for x in events[event][sample]:
        xvals.append(x)
        yvals.append(y + theme[event].get('ydelta', 0))
    ax.scatter(xvals, yvals, c=theme[event]['color'], s=theme[event]['markersize'], marker=theme[event]['marker'], label=event)
  # xvals = event_value
  # yvals = sample y_pos
  # cvals = color
  # marker = marker

  # indicator scatter
  for theme_name in theme:
    if not theme_name.startswith('indicator_'):
      continue
    logging.info(theme_name)
    name = theme_name.replace('indicator_', '')
    xvals = []
    yvals = []
    for y, sample in enumerate(labels):
      if sample not in indicator:
        continue
      if indicator[sample] == name:
        yvals.append(y)
        if start is None:
          xvals.append(-1)
        else:
          xvals.append(start-1)
    if len(xvals) > 0:
      ax.scatter(xvals, yvals, c=theme[theme_name]['color'], s=theme[theme_name]['markersize'], marker=theme[theme_name]['marker'], label=name)

  ax.legend(frameon=True)
  ax.set_xlabel('Age (years)')
  ax.set_ylabel('Individual')

  plt.tight_layout()
  plt.savefig(target, dpi=dpi, transparent=False) #plotme.settings.TRANSPARENT)
  matplotlib.pyplot.close('all')
  logging.info('done')

  #fig, ax=plt.subplots()
  #plt.subplots_adjust( top=0.9, bottom=0.1)
  #df.sort_values('TRTDURM', inplace=True)
  #ax.barh('SUBJID', 'TRTDURM', color="gray", zorder=-1 , data=df)

  #ax.scatter('DTHDM','SUBJID', data=df, marker='*',color='red', s=30, zorder=1, label='Death')
  #ax.scatter('TRTDURM','SUBJID', data=ongo, marker='o',color='green', s=30, zorder=1, label='Ongoing')
  #ax.scatter('ADM','SUBJID', data=cr, marker='v',color='blue', s=30, zorder=1, label='CR')

  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Make a swimmer plot')
  parser.add_argument('--data', required=True, help='input')
  parser.add_argument('--start', required=False, type=int, help='starting age')
  parser.add_argument('--target', required=False, default='plot.png', help='target filename')
  parser.add_argument('--width', required=False, default=12, type=float, help='figsize width')
  parser.add_argument('--dpi', required=False, default=72, type=int, help='figsize width')
  parser.add_argument('--height', required=False, default=18, type=int, help='fontsize')
  parser.add_argument('--indicator', required=True, help='indicator col name')
  parser.add_argument('--bar_indicator', required=False, help='bar indicator col name')
  parser.add_argument('--sort_key', required=False, help='indicator sort')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.data, args.target, width=args.width, height=args.height, dpi=args.dpi, start=args.start, indicator_col=args.indicator, bar_indicator=args.bar_indicator, sort_key=args.sort_key)
