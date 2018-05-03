"""
Code to validate predictions.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def score_preds(preds, df=None, y=None):
  if df is not None:
    y = df['emVar_Hit']
  else:
    assert y is not None
  roc_score = roc_auc_score(y, preds)
  auprc_score = average_precision_score(y, preds)
  print('AUROC:\t{}\tAUPRC:\t{}'.format(roc_score, auprc_score))
  return roc_score, auprc_score


def pr(cvdf_chunk, col='PredValue'):
  """Calculate the precision and recall."""
  precision, recall, thresholds = \
      precision_recall_curve(cvdf_chunk['class'].abs(), cvdf_chunk[col].abs())
  auprc = auc(recall, precision)
  return precision, recall, thresholds, auprc


def pr_one_against_many(preds):
  """
  Calculate the precision and recall curves for the following outcomes:

    - called variant (direction != 0) against uncalled (direction == 0)
    - positive variant (direction ==  1) against uncalled or negative
    - negative variant (direction == -1) against uncalled or positive
  """
  N = preds.shape[0]
  #
  # Which samples have particular classes
  #
  positive = preds['class'] ==  1
  negative = preds['class'] == -1
  uncalled = preds['class'] ==  0
  #
  # Double check everything is as expected
  #
  assert np.all(~uncalled == positive | negative)
  assert np.all(uncalled | positive | negative)
  assert uncalled.sum() + positive.sum() + negative.sum() == N
  #
  # Which samples are predicted to have particular classes
  #
  pred_positive = preds['Direction'] ==  1
  pred_negative = preds['Direction'] == -1
  pred_uncalled = preds['Direction'] ==  0
  #
  # Called against uncalled
  #
  scores = np.empty(N)
  scores[pred_positive] = preds.loc[pred_positive, 'P_Direction']
  scores[pred_negative] = preds.loc[pred_negative, 'P_Direction']
  scores[pred_uncalled] = - preds.loc[pred_uncalled, 'P_Direction']
  precision, recall, thresholds = precision_recall_curve(~uncalled, scores)
  auprc_called_uncalled = auc(recall, precision)
  #
  # Positive against rest
  #
  scores = np.empty(N)
  scores[pred_positive] = preds.loc[pred_positive, 'P_Direction']
  scores[pred_negative] = - preds.loc[pred_negative, 'P_Direction']
  scores[pred_uncalled] = - preds.loc[pred_uncalled, 'P_Direction']
  precision, recall, thresholds = precision_recall_curve(positive, scores)
  auprc_positive_rest = auc(recall, precision)
  #
  # Negative against rest
  #
  scores = np.empty(N)
  scores[pred_positive] = - preds.loc[pred_positive, 'P_Direction']
  scores[pred_negative] = preds.loc[pred_negative, 'P_Direction']
  scores[pred_uncalled] = - preds.loc[pred_uncalled, 'P_Direction']
  precision, recall, thresholds = precision_recall_curve(negative, scores)
  auprc_negative_rest = auc(recall, precision)
  #
  return auprc_called_uncalled, auprc_positive_rest, auprc_negative_rest
