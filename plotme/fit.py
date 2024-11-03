#!/usr/bin/env python
'''
  fit a distribution to a list of points
'''

import argparse
import logging
import sys

import numpy as np

from scipy.stats import beta

def main(xs, distribution='normal'):
  logging.info('starting...')

  if distribution == 'normal':
    mean = np.mean(xs)
    sd = np.std(xs)
    low = mean - 2 * sd
    high = mean + 2 * sd
    sys.stdout.write('Dist\t{}\nMean\t{:.6f}\nSD\t{:.6f}\nCI_Low\t{:.6f}\nCI_High\t{:.6f}\n'.format(distribution, mean, sd, low, high))
  elif distribution == 'lognormal':
    log_xs = np.log(xs)
    log_mean = np.mean(log_xs)
    log_sd = np.std(log_xs)
    log_low = log_mean - 2 * log_sd
    log_high = log_mean + 2 * log_sd
    sys.stdout.write('Dist\t{}\nMean\t{:.6f}\nSD\t{:.6f}\nCI_Low\t{:.6f}\nCI_High\t{:.6f}\n'.format(distribution, np.exp(log_mean), np.exp(log_sd), np.exp(log_low), np.exp(log_high)))
  elif distribution == 'beta':
    mean = np.mean(xs)
    sd = np.std(xs)
    n = mean * (1 - mean) / (sd ** 2)
    a = mean * n
    b = (1 - mean) * n
    ci_low, ci_high = beta.ppf([0.025, 0.975], a, b)
    sys.stdout.write('Dist\t{}\na\t{:.6f}\nb\t{:.6f}\nCI_Low\t{:.6f}\nCI_High\t{:.6f}\n'.format(distribution, a, b, ci_low, ci_high))
 
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Assess MSI')
  parser.add_argument('--xs', required=True, nargs='+', type=float, help='points to fit')
  parser.add_argument('--distribution', required=False, default='normal', help='distribution [normal, lognormal, beta]')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.xs, args.distribution)

