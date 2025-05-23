#!/usr/bin/env python

import argparse
import csv
import logging
import math
import os
import random
import sys

import matplotlib
# turn this off for show
# matplotlib.use('Agg')
import matplotlib.colors 
import matplotlib.pyplot as plt

from pylab import rcParams

#import numpy.polynomial
import numpy
import numpy.random
import scipy.stats
import scipy.interpolate 
import plotme.settings

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS = ('^', 'x', 'v', 'o', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '.', ',', '|', '_')
CMAP_DEFAULT= (0.6, 0.6, 0.6, 0.5)  # non-numeric => black

# based on https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762#53865762
def density_scatter( x, y, color, fig=None, ax=None, sort=True, bins=10, ranges=None, cutoff=None, resolution=100, markersize=None, opacity=0.5, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
      fig, ax = plt.subplots()

    # histogram method
    data, x_e, y_e = numpy.histogram2d(x, y, bins=bins, density=True)

    # expand
    expanded = []
    expanded.append([0] * (len(data[0]) + 2))
    for row in data:
      expanded.append([0] + list(row) + [0])
    expanded.append([0] * (len(data[0]) + 2))
    
    diff = x_e[1] - x_e[0]
    x_e = numpy.insert(x_e, 0, x_e[0] - diff)
    x_e = numpy.append(x_e, [x_e[-1] + diff])
    diff = y_e[1] - y_e[0]
    y_e = numpy.insert(y_e, 0, y_e[0] - diff)
    y_e = numpy.append(y_e, [y_e[-1] + diff])

    stepx = (ranges[1]-ranges[0])/resolution
    stepy = (ranges[3]-ranges[2])/resolution
    new_xs, new_ys = numpy.mgrid[ranges[0]-stepx:ranges[1]+stepx:stepx, ranges[2]-stepy:ranges[3]+stepy:stepy]
    new_xs = new_xs.flatten()
    new_ys = new_ys.flatten()

    z = scipy.interpolate.interpn( (0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ), expanded, numpy.vstack([new_xs, new_ys]).T, method="linear", bounds_error = False)
    #z = scipy.interpolate.interpn( (0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ), data, numpy.vstack([new_xs, new_ys]).T, method="linear", bounds_error = False)
    z[numpy.where(numpy.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
      xs = []
      ys = []
      zs = []
      for idx in z.argsort():
        xs.append(new_xs[idx])
        ys.append(new_ys[idx])
        zs.append(z[idx])
      new_xs, new_ys, z = xs, ys, zs
    znorm = matplotlib.colors.Normalize(vmin = numpy.min(z), vmax = numpy.max(z))
    z = znorm(z)

    zs = []
    mx = (None, None)
    for n, i in enumerate(z):
      if i < cutoff:
        zs.append(0)
      else:
        zs.append(i * opacity)
    markersize = markersize or 20 * (100/resolution) * (100/resolution)
    ax.scatter(new_xs, new_ys, c=color, zorder=0, s=markersize, alpha=zs, marker='s', **kwargs)
    return ax

def plot_scatter(data_fh, target, xlabel, ylabel, zlabel, figsize=12, fontsize=18, x_log=False, y_log=False, title=None, x_label=None, y_label=None, wiggle=0, delimiter='\t', z_color=None, z_color_map=None, label=None, join=False, y_annot=None, x_annot=None, dpi=72, markersize=20, z_cmap=None, x_squiggem=0.005, y_squiggem=0.005, marker='o', lines=[], line_of_best_fit=False, line_of_best_fit_by_category=False, projectionlabel=None, projectionview=None, include_zero=False, max_x=None, max_y=None, skip=True, poly=None, density=False, density_bins=10, density_cutoff=0.4, density_resolution=100, density_markersize=None, density_opacity=0.5):
  logging.info('starting...')
  try:
    matplotlib.style.use('seaborn-v0_8')
  except:
    matplotlib.style.use('seaborn')

  included = total = 0
  xvals = []
  yvals = []
  zvals = []
  if projectionlabel is not None:
    projection = []
  else:
    projection = 0
  cvals = []
  mvals = []
  lvals = []

  zvals_seen = []
  markers_seen = set()
  colors_seen = set()
  zvals_range = (1e99, -1e99)

  for row in csv.DictReader(data_fh, delimiter=delimiter):
    try:
      if skip and row[xlabel] == '' or row[ylabel] == '':
        continue
      included += 1
      xval = float(row[xlabel]) + (numpy.random.normal() - 0.5) * 2 * wiggle # x axis value
      yval = float(row[ylabel]) + (numpy.random.normal() - 0.5) * 2 * wiggle # y axis value
      xvals.append(xval)
      yvals.append(yval)
      if projectionlabel is not None:
        projection.append(float(row[projectionlabel]) + (numpy.random.normal() - 0.5) * 2 * wiggle)
      # process z
      if zlabel is not None:
        if row[zlabel] not in zvals_seen and z_cmap is None:
          zvals_seen.append(row[zlabel])

        z_color_map_found = False
        if z_color_map is not None: # directly map z values to a colour
          for m in z_color_map:
            logging.debug('splitting %s', m)
            name, value = m.rsplit(':', 1)
            logging.debug('comparing %s to %s', name, row[zlabel])
            if name == row[zlabel]:
              color, marker = value.split('/')
              cvals.append(color)
              colors_seen.add(color)
              mvals.append(marker)
              markers_seen.add(marker)
              z_color_map_found = True
              logging.debug('marker for %s added', name)
              break

        if z_color and not z_color_map_found and z_cmap is None: # use a predefined list of distinct colours
          ix = zvals_seen.index(row[zlabel])
          cvals.append(COLORS[ix % len(COLORS)])
          colors_seen.add(COLORS[ix % len(COLORS)])
          jx = int(ix / len(COLORS))
          mvals.append(MARKERS[jx % len(MARKERS)])
          markers_seen.add(MARKERS[jx % len(MARKERS)])

        if z_cmap is not None:
          try:
            zvals_range = (min((float(row[zlabel]), zvals_range[0])), max((float(row[zlabel]), zvals_range[1])))
          except ValueError:
            pass # skip non-numeric

        zvals.append(row[zlabel])

      if label is not None:
        lvals.append(row[label].replace('/', '\n'))

    except:
      logging.warning('Failed to include (is %s numeric?) %s', zlabel, row)
      raise

    total += 1

  # assign continuous color if z_cmap
  if z_cmap is not None:
    logging.info('cmap has range %s', zvals_range)
    cmap = matplotlib.cm.get_cmap(z_cmap)
    norm = matplotlib.colors.Normalize(vmin=zvals_range[0], vmax=zvals_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cvals = []
    for x in zvals:
      try:
        cvals.append(m.to_rgba(float(x)))
      except ValueError:
        cvals.append(CMAP_DEFAULT)
    logging.debug(cvals)

  logging.info('finished reading %i of %i records', included, total)

  if len(xvals) == 0:
    logging.warning('No data to plot')
    return

  matplotlib.rcParams.update({'font.size': fontsize})
  fig = plt.figure(figsize=(figsize, 1 + int(figsize * len(yvals) / len(xvals))))
  if projectionlabel is None:
    ax = fig.add_subplot(111)
  else:
    ax = fig.add_subplot(111, projection='3d')
    if projectionview is not None and len(projectionview) > 0:
      elev, azim, roll = [int(x) for x in projectionview]
      ax.view_init(elev=elev, azim=azim, roll=roll)

  if y_label is None:
    ax.set_ylabel(ylabel)
  else:
    logging.debug('y_label is %s', y_label)
    ax.set_ylabel(y_label)

  if x_label is None:
    ax.set_xlabel(xlabel)
  else:
    logging.debug('x_label is %s', x_label)
    ax.set_xlabel(x_label)

  if projectionlabel is not None:
    ax.set_zlabel(projectionlabel)

  if z_color or z_color_map is not None:
    for zval in zvals_seen:
      if projectionlabel is None:
        vals = [list(x) for x in zip(xvals, yvals, zvals, cvals, mvals) if x[2] == zval]
      else:
        vals = [list(x) for x in zip(xvals, yvals, zvals, cvals, mvals, projection) if x[2] == zval]
      marker = vals[0][4]
      if projectionlabel is None:
        if join:
          ax.plot([x[0] for x in vals], [x[1] for x in vals], c=[x[3] for x in vals], markersize=markersize, marker=marker, label=zval, alpha=0.8)
        else:
          ax.scatter([x[0] for x in vals], [x[1] for x in vals], c=[x[3] for x in vals], s=markersize, marker=marker, label=zval, alpha=0.8)
      else:
        if join:
          ax.plot([x[0] for x in vals], [x[1] for x in vals], c=vals[0][3], markersize=markersize, marker=marker, label=zval, alpha=0.8)
        else:
          ax.scatter([x[0] for x in vals], [x[1] for x in vals], zs=[x[5] for x in vals], c=[x[3] for x in vals], s=markersize, marker=marker, label=zval, alpha=0.8)
      ax.legend()
      #if join: # TODO does this work?
      #  ax.join([x[0] for x in vals], [x[1] for x in vals], c=[x[3] for x in vals], marker=vals[0][4], label=zval, alpha=0.8)
  elif z_cmap is not None:
    #logging.info('plotting %s %s %s %s %s', xvals, yvals, cvals, markersize, marker)
    if projectionlabel is None:
      ax.scatter(xvals, yvals, c=cvals, s=markersize, marker=marker)
    else:
      ax.scatter(xvals, yvals, zs=projection, c=cvals, s=markersize, marker=marker)
    #cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.5)
    ax.figure.colorbar(m, ax=ax, label=zlabel, fraction=0.04, pad=0.01, shrink=0.5)
  else:
    if projectionlabel is None:
      ax.scatter(xvals, yvals, s=markersize, marker=marker)
    else:
      ax.scatter(xvals, yvals, zs=projection, s=markersize, marker=marker)
    if join:
      ax.plot(xvals, yvals)

  if line_of_best_fit is not None or poly is not None or density is not None:
    safe_vals = [xy for xy in zip(xvals, yvals) if not math.isnan(xy[1])]
    safe_xvals = [xy[0] for xy in safe_vals]
    safe_yvals = [xy[1] for xy in safe_vals]

  if line_of_best_fit:
    res = scipy.stats.linregress(safe_xvals, safe_yvals)
    logging.debug('xvals: %s res: %s', safe_xvals, res)
    yvals_res = [res.intercept + res.slope * xval for xval in safe_xvals]
    correlation = scipy.stats.pearsonr(safe_xvals, safe_yvals)
    ax.plot(safe_xvals, yvals_res, color='orange', label='correlation {:.3f}\ngradient {:.3f}\npvalue {:.3f}'.format(correlation[0], res.slope, correlation[1]), linewidth=1)
    ax.legend()

  if line_of_best_fit_by_category:
    for zval in zvals_seen:
      vals = [list(x) for x in zip(xvals, yvals, zvals, cvals, mvals) if x[2] == zval]
      res = scipy.stats.linregress([x[0] for x in vals], [x[1] for x in vals])
      yvals_res = [res.intercept + res.slope * xval for xval in xvals]
      correlation = scipy.stats.pearsonr([x[0] for x in vals], [x[1] for x in vals])
      ax.plot(xvals, yvals_res, color=vals[0][3], label='{} correlation {:.3f}\ngradient {:.3f}\npvalue {:.3f}'.format(zval, correlation[0], res.slope, correlation[1]), linewidth=1)
      ax.legend()

  if poly is not None:
    res = numpy.polyfit(safe_xvals, safe_yvals, poly)
    p = numpy.poly1d(res)
    xy = sorted(zip(safe_xvals, safe_yvals))
    ax.plot([v[0] for v in xy], [p(v[0]) for v in xy], color='purple', label='polyfit degree {}'.format(poly), linewidth=1)
    ax.legend()

  if density: # assume by category
    for idx, zval in enumerate(zvals_seen):
      vals = [list(x) for x in zip(xvals, yvals, zvals, cvals, mvals) if x[2] == zval]
      #if idx == 1:
      density_scatter([x[0] for x in vals], [x[1] for x in vals], color=vals[0][3], fig=fig, ax=ax, sort=False, bins=density_bins, ranges=(min(safe_xvals), max(safe_xvals), min(safe_yvals), max(safe_yvals)), cutoff=density_cutoff, resolution=density_resolution, markersize=density_markersize, opacity=density_opacity)

  if zlabel is not None:
    if not z_color and not z_cmap:
      for x, y, z in zip(xvals, yvals, zvals):
        ax.annotate(z, (x, y), fontsize=fontsize)

  # alternative labelling
  if label is not None:
    for x, y, z in zip(xvals, yvals, lvals):
      ax.annotate(z, (x, y), fontsize=fontsize)

  if y_annot is not None:
    for ya in y_annot:
      color = 'red'
      if ':' in ya:
        ya, color = ya.split(':')
      label, height = ya.split('=')
      logging.debug('labelling line at %s with %s', height, label)
      ax.axhline(float(height), color=color, linewidth=1)
      ax.annotate(label, (min(xvals), float(height) + y_squiggem), fontsize=8)

  if x_annot is not None:
    color = 'red'
    for xa in x_annot:
      if ':' in xa:
        xa, color = xa.split(':')
      label, width = xa.split('=')
      logging.debug('labelling line at %s with %s', width, label)
      ax.axvline(float(width), color='red', linewidth=1)
      ax.annotate(label, (float(width) + x_squiggem, min(yvals)), fontsize=8)

  if lines is not None:
    for line in lines:
      x1, y1, x2, y2, c = line.split(',')
      ax.plot([float(x1), float(x2)], [float(y1), float(y2)], color=c, marker='')

  if title is not None:
    ax.set_title(title)

  if y_log: # does this work?
    ax.set_yscale('log')
  if x_log: # does this work?
    ax.set_xscale('log')

  if include_zero:
    ax.set_xlim(left=min([-wiggle] + xvals))
    ax.set_ylim(bottom=min([-wiggle] + yvals))
  if max_x is not None:
    ax.set_xlim(right=max([max_x] + xvals))
  if max_y is not None:
    ax.set_ylim(top=max([max_y] + yvals))

  logging.info('done processing %i of %i. saving to %s...', included, total, target)
  plt.tight_layout()
  if target == 'show':
    plt.show()
  else:
    plt.savefig(target, dpi=dpi, transparent=False) #plotme.settings.TRANSPARENT)
  matplotlib.pyplot.close('all')
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Scatter plot')
  parser.add_argument('--x', required=True, help='x column name')
  parser.add_argument('--y', required=True, help='y column name')
  parser.add_argument('--z', required=False, help='z column name (colour)')
  parser.add_argument('--projection', required=False, help='additional column for 3d projection')
  parser.add_argument('--projection_view', required=False, nargs='+', help='view point as 3 integers elev azim roll')
  parser.add_argument('--label', required=False, help='label column')
  parser.add_argument('--z_color', action='store_true', help='use colours for z')
  parser.add_argument('--z_color_map', required=False, nargs='+', help='specify color/marker for z: label:color/marker')
  parser.add_argument('--z_cmap', required=False, help='z is continuous and use a color map')
  parser.add_argument('--title', required=False, help='z column name')
  parser.add_argument('--x_label', required=False, help='label on x axis')
  parser.add_argument('--y_label', required=False, help='label on y axis')
  parser.add_argument('--figsize', required=False, default=12, type=float, help='figsize width')
  parser.add_argument('--fontsize', required=False, default=18, type=int, help='fontsize')
  parser.add_argument('--markersize', required=False, default=20, type=int, help='fontsize')
  parser.add_argument('--marker', required=False, default='o', help='default marker')
  parser.add_argument('--dpi', required=False, default=plotme.settings.DPI, type=int, help='dpi')
  parser.add_argument('--wiggle', required=False, default=0, type=float, help='randomly perturb data')
  parser.add_argument('--x_squiggem', required=False, default=0.005, type=float, help='offset for text')
  parser.add_argument('--y_squiggem', required=False, default=0.005, type=float, help='offset for text')
  parser.add_argument('--delimiter', required=False, default='\t', help='input file delimiter')
  parser.add_argument('--x_log', action='store_true', help='log xy')
  parser.add_argument('--y_log', action='store_true', help='log xy')
  parser.add_argument('--join', action='store_true', help='join points')
  parser.add_argument('--y_annot', required=False, nargs='*', help='add horizontal lines of the form label=height')
  parser.add_argument('--x_annot', required=False, nargs='*', help='add vertical lines of the form label=height')
  parser.add_argument('--lines', required=False, nargs='*', help='add unannotated lines of the form x1,y1,x2,y2,color')
  parser.add_argument('--line_of_best_fit', action='store_true', help='include line of best fit for entire dataset')
  parser.add_argument('--line_of_best_fit_by_category', action='store_true', help='include line of best fit for each z')
  parser.add_argument('--polyfit', type=int, required=False, help='polyfit with degree n')
  parser.add_argument('--animate', action='store_true', help='animate 3d plot (requires ffmpeg)')
  parser.add_argument('--include_zero', action='store_true', help='include zero o both axes')
  parser.add_argument('--max_x', type=float, required=False, help='include this value in x')
  parser.add_argument('--max_y', type=float, required=False, help='include this value in y')
  parser.add_argument('--density', action='store_true', help='add a density overlay')
  parser.add_argument('--density_bins', required=False, default=10, type=int, help='add a density overlay')
  parser.add_argument('--density_cutoff', required=False, default=0.0, type=float, help='add a density overlay')
  parser.add_argument('--density_resolution', required=False, default=100, type=int, help='add a density overlay')
  parser.add_argument('--density_markersize', required=False, type=int, help='add a density overlay')
  parser.add_argument('--density_opacity', required=False, type=float, default=0.5, help='opacity of density overlay')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--target', required=False, default='plot.png', help='plot filename')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  if args.animate:
    n = 0
    ifh = sys.stdin.readlines()
    for x in range(1, 89, 1):
      v = [20, x, 0]
      logging.info('frame %i', n)
      plot_scatter(ifh, 'anim-{:02d}.png'.format(n), args.x, args.y, args.z, args.figsize, args.fontsize, args.x_log, args.y_log, args.title, args.x_label, args.y_label, args.wiggle, args.delimiter, args.z_color, args.z_color_map, args.label, args.join, args.y_annot, args.x_annot, args.dpi, args.markersize, args.z_cmap, args.x_squiggem, args.y_squiggem, args.marker, args.lines, args.line_of_best_fit, args.line_of_best_fit_by_category, args.projection, v, poly=args.poly)
      n += 1      
    # make animation
    os.system('ffmpeg -r 4 -i anim-%02d.png -vcodec libx264 -acodec aac {}.mp4'.format(args.target))
  else:
    plot_scatter(sys.stdin, args.target, args.x, args.y, args.z, args.figsize, args.fontsize, args.x_log, args.y_log, args.title, args.x_label, args.y_label, args.wiggle, args.delimiter, args.z_color, args.z_color_map, args.label, args.join, args.y_annot, args.x_annot, args.dpi, args.markersize, args.z_cmap, args.x_squiggem, args.y_squiggem, args.marker, args.lines, args.line_of_best_fit, args.line_of_best_fit_by_category, args.projection, args.projection_view, args.include_zero, args.max_x, args.max_y, poly=args.polyfit, density=args.density, density_bins=args.density_bins, density_cutoff=args.density_cutoff, density_resolution=args.density_resolution, density_markersize=args.density_markersize, density_opacity=args.density_opacity)
