"""
Spatial smoothing code.
"""

from functools import partial
import numpy as np
import pandas as pd


RE = 'regulatory_element'

def maxmag(x):
  "The value of maximum magnitude."
  return x.values[x.abs().values.argmax()]


def maxabs(x):
  "The maximum absolute value."
  return np.max(np.abs(x))


def meanabs(x):
  "The mean absolute value."
  return np.mean(np.abs(x))


def medianabs(x):
  "The median absolute value."
  return np.median(np.abs(x))


def smooth_chunk(v, lengthscale, zero_diag):
  "Smooth a vector of arbitrary length."
  N = len(v)
  K = rbf_kernel(N, lengthscale, zero_diag)
  result = K @ v.as_matrix()
  assert(1 == result.ndim)
  assert(N == result.shape[0])
  return result


def rbf_kernel(N, lengthscale, zero_diag=False):
  """
  Create a RBF kernel smoothing matrix.
  """
  K = np.empty((N, N)) 
  for n in range(N):
    dists = np.arange(-n, N-n)
    K[n] = np.exp(- dists**2 / lengthscale**2)
  if zero_diag:
    np.fill_diagonal(K, 0)
  K /= K.sum(axis=0)
  return K


def smooth_df(preds, aggfn=medianabs, lengthscale=2):
  byelem = \
      preds. \
      groupby((RE, 'chunk_id', 'Pos')). \
      agg({'Value': aggfn, 'PredValue': aggfn}). \
      rename(columns={'PredValue': 'PredValueAgg', 'Value': 'ValueAgg'})
  smoothed = \
      byelem.groupby(level=(0, 1)). \
      transform(partial(smooth_chunk, lengthscale=lengthscale, zero_diag=True)). \
      rename(columns={
          'PredValueAgg': 'PredSmoothedAgg',
          'ValueAgg': 'SmoothedAgg'})
  return smoothed.merge(byelem, left_index=True, right_index=True)

