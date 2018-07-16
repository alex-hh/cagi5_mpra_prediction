import os
import multiprocessing
import pickle
import re
from functools import partial
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
from pandas.api.types import CategoricalDtype
import xgboost as xgb
import lightgbm as lgbm
import catboost
from constants import MINP
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from utils import snp_feats_from_preds, encode_sequences
from cagi5_utils import get_breakpoint_df, get_chunk_counts



class Classifier(object):
  """
  Classifies SNVs using logistic regression or XGBoost.
  """

  def __init__(
      self,
      features,
      model_name='lr',
      multiclass='ovr',
      model_kwargs={},
      verbose=False):
    self.features = features
    self.model_kwargs = model_kwargs
    self.model_name = model_name
    self.multiclass=multiclass
    self.verbose = verbose

  def fit(self, X, y):
    # In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    # scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    # entropy loss if the 'multi_class' option is set to 'multinomial'
    if self.model_name == 'lr':
      self.model = LogisticRegression(penalty='l2', C=0.01, multi_class=self.multiclass)
    elif self.model_name == 'xgb':
      self.model = xgb.XGBClassifier(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
    else:
      return ValueError('Unknown model name: {}'.format(self.model_name))
    sample_weight = compute_sample_weight('balanced', y) # not sure if classes need to be labelled 0,1,2 (if so can use label encoder)
    self.model.fit(X, y, sample_weight=sample_weight)
    if self.verbose:
      print('Train accuracy: {}'.format(self.model.score(X, y)))

  def predicted_columns(self):
    return ['PredClass']

  def predict(self, X, index):
    return pd.DataFrame({'PredClass': self.model.predict(X)}, index=index)

  def get_features(self, df, elem=None):
    return self.features.get_features(df, elem)

  def get_response(self, df):
    return df['class']



class Regression(object):
  """
  Regresses confidence and effect size using XGBoost.
  """

  def __init__(
      self,
      features,
      model_name='xgb',
      model_kwargs={},
      verbose=False):
    self.features = features
    self.model_kwargs = model_kwargs
    self.model_name = model_name
    self.verbose = verbose

  def fit(self, X, y):
    if self.model_name == 'xgb':
      self.model_value = xgb.XGBRegressor(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
      self.model_conf = xgb.XGBRegressor(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
    elif self.model_name == 'lgbm':
      self.model_value = lgbm.LGBMRegressor(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
      self.model_conf = lgbm.LGBMRegressor(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
    elif self.model_name == 'catboost':
      self.model_value = catboost.CatBoostRegressor(**self.model_kwargs, logging_level='Silent')
      self.model_conf = catboost.CatBoostRegressor(**self.model_kwargs, logging_level='Silent')
    elif self.model_name == 'elasticnet':
      self.model_value = ElasticNetCV(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
      self.model_conf = ElasticNetCV(**self.model_kwargs, n_jobs=multiprocessing.cpu_count() - 1)
    else:
      return ValueError('Unknown model name: {}'.format(self.model_name))
    self.model_value.fit(X, y['Value'])
    self.model_conf.fit(X, y['Confidence'])
    if self.verbose:
      print('Train accuracy (value): {}'.format(self.model_value.score(X, y)))
      print('Train accuracy (confidence): {}'.format(self.model_conf.score(X, y)))

  def predicted_columns(self):
    return ['PredValue', 'PredConfidence']

  def predict(self, X, index):
    return pd.DataFrame(
      data={
        'PredValue': self.model_value.predict(X),
        'PredConfidence': self.model_conf.predict(X)
      },
      index=index)

  def get_features(self, df, elem=None):
    return self.features.get_features(df, elem)

  def get_response(self, df):
    return df[['Value', 'Confidence']]

  @staticmethod
  def make_submission(preds):
    """
    Take the output from a Regression model and call variants.

    I.e. estimate each variant's direction, probability of
    correct direction estimation, confidence score and s.e.
    of the confidence score.
    """
    N = preds.shape[0]
    direction = np.zeros(N, dtype=int)
    direction[preds['PredValue'] >= .1] = 1
    direction[preds['PredValue'] <= -.1] = -1
    preds['Direction'] = direction
    # Rank the predicted values, we use the ranks to estimate the
    # probabilities our direction estimate is correct
    ranks = scipy.stats.rankdata(preds['PredValue'].abs())
    preds['ValueRank'] = ranks
    # What is the highest rank of the uncalled variants (class == 0)?
    uncalled = direction == 0
    # Estimate the probability of a correct direction assignment
    # Scale the ranks to between .5 and 1 for called and separately uncalled variants
    scaler = partial(linear_scale, lower=MINP, upper=1. - 1e-7)
    p_direction = np.empty_like(ranks)
    p_direction[uncalled] = scaler(-ranks[uncalled])
    p_direction[~uncalled] = scaler(ranks[~uncalled])
    preds['P_Direction'] = np.clip(p_direction, 1e-5, 1 - 1e-5)
    # Use the predicted confidences
    preds['Confidence'] = np.clip(preds['PredConfidence'], 1e-5, 1 - 1e-5)
    # Use any old standard error - not sure how to estimate this without an ensemble.
    preds['SE'] = .1
    return preds


class RegressionClassifier(object):
  """
  Fits regression models to Value and Confidence and a classifier
  for Direction.
  """

  def __init__(
      self,
      regression,
      classifier,
      verbose=False):
    self.regression = regression
    self.classifier = classifier
    self.verbose = verbose

  def fit(self, X, y):
    #
    # Fit the regression to the values and the confidences
    self.regression.fit(X, y.iloc[:, :2])
    #
    # fit the classifier to the class with the augmented features
    self.classifier.fit(X, y.iloc[:, 2:])

  def _augment_X(self, X):
    #
    # get the predicted values and confidences
    pred_values, pred_confs = self._regression_predict(X)
    #
    # augment the features with the predictions
    return (
        np.concatenate(
          (X, np.expand_dims(pred_values, axis=1), np.expand_dims(pred_confs, axis=1)),
          axis=1),
        pred_values,
        pred_confs)

  def _regression_predict(self, X):
    pred_values = self.regression.model_value.predict(X)
    pred_confs = self.regression.model_conf.predict(X)
    return pred_values, pred_confs

  def predicted_columns(self):
    return ['PredValue', 'PredConfidence', 'PredClass', 'NegativeProb', 'UncalledProb', 'PositiveProb']

  def predict(self, X, index):
    #
    # augment the features with the regression predictions of values and confidences
    pred_values, pred_confs = self._regression_predict(X)
    pred_class = self.classifier.model.predict(X)
    pred_classprobs = self.classifier.model.predict_proba(X)
    return pd.DataFrame({
        'PredValue': pred_values,
        'PredConfidence': pred_confs,
        'PredClass': pred_class,
        'NegativeProb': pred_classprobs[:, 0],
        'UncalledProb': pred_classprobs[:, 1],
        'PositiveProb': pred_classprobs[:, 2],
      },
      index=index)

  def get_features(self, df, elem=None):
    return self.regression.get_features(df, elem)

  def get_response(self, df):
    return df[['Value', 'Confidence', 'class']]

  @staticmethod
  def make_submission(preds):
    """
    Take the output from this model and call variants.

    I.e. estimate each variant's direction, probability of
    correct direction estimation, confidence score and s.e.
    of the confidence score.
    """
    N = preds.shape[0]
    preds['Direction'] = preds['PredClass']
    #
    # Which samples were classified into each class?
    neg_idx = preds['Direction'] == -1
    unc_idx = preds['Direction'] ==  0
    pos_idx = preds['Direction'] ==  1
    #
    # Collate the probabilities that the class predictions are correct
    p_direction = np.zeros(N)
    p_direction[neg_idx] = preds.loc[neg_idx, 'NegativeProb']
    p_direction[unc_idx] = preds.loc[unc_idx, 'UncalledProb']
    p_direction[pos_idx] = preds.loc[pos_idx, 'PositiveProb']
    assert np.all(p_direction > .33)  # Check our classifier chose a popular class
    preds['P_Direction'] = p_direction
    # Use the predicted confidences
    preds['Confidence'] = np.clip(preds['PredConfidence'], 1e-5, 1 - 1e-5)
    # Use any old standard error - not sure how to estimate this without an ensemble.
    preds['SE'] = .1
    #
    # Calculate continuous scores
    predictions['ContinuousScore'] = np.inf
    predictions.loc[neg_idx, 'ContinuousScore'] = - 2 / 3 - predictions.loc[neg_idx, 'NegativeProb']
    predictions.loc[unc_idx, 'ContinuousScore'] = \
        3 / 2 * (predictions.loc[unc_idx, 'PositiveProb'] - predictions.loc[unc_idx, 'NegativeProb'])
    predictions.loc[pos_idx, 'ContinuousScore'] = 2 / 3  + predictions.loc[pos_idx, 'PositiveProb']
    return preds




def linear_scale(x, lower=0, upper=1):
  """
  Transform x linearly into the range [lower, upper]
  """
  xmin = x.min()
  xmax = x.max()
  return lower + (upper - lower) * (x - xmin) / (xmax - xmin)


class Features(ABC):
  """
  Abstract base class for features.
  """

  @abstractmethod
  def get_features(self, df, elem=None):
    pass


class DeepSeaSNP(Features):

  def __init__(
      self,
      use_saved_preds=True,
      feattypes=['diff'],
      filename_fmt='deepsea_{}_preds.npy'):
    self.use_saved_preds = use_saved_preds
    self.feattypes = feattypes
    self.filename_fmt = filename_fmt

  def _npy_path(self, variant):
    return os.path.join('data', 'cagi5_mpra', self.filename_fmt.format(variant))

  def get_features(self, df, elem=None):
    if self.use_saved_preds:
      train_inds = df.index.values
      train_ref = np.load(self._npy_path('ref'))[train_inds]
      train_alt = np.load(self._npy_path('alt'))[train_inds]

    return snp_feats_from_preds(train_ref, train_alt, self.feattypes)


class Stacked(Features):
  """
  Stacked features derived from a model cross-validated across all training data
  """

  def __init__(self, tag='deep-e1h-dnase-cons'):
    self.feats = np.load('data/stacked-{}.npy'.format(tag))

  def get_features(self, df, elem=None):
    return self.feats[df.index.values]


class DNase(Features):
  """
  Sequence features derived from downloaded DNase tracks.
  """

  def __init__(self, idxs=None, test=False):
    """
    Here idxs are not idxs of data frame rows but idxs of DNase features.
    """
    if test:
      path = 'data/dnase-features-test.npy'
    else:
      path = 'data/dnase-features.npy'
    self.feats = np.load(path)
    if idxs is not None:
      self.feats = self.feats[:, idxs]

  def get_features(self, df, elem=None):
    return self.feats[df.index.values]


class DSDataKerasModel(Features):

  def __init__(self, experiment_name, feattypes=['diff'], alllayers=False, layers=[],
               filesuffix=''):
    self.feattypes = feattypes
    self.experiment_name = experiment_name
    self.layers = layers
    self.alllayers = alllayers
    self.filesuffix = filesuffix

  def get_refalt_preds(self, df, seqlen=1000, inds=None):
    assert 'ref_sequence' in df.columns
    ref_onehot = encode_sequences(df['ref_sequence'], seqlen=seqlen, inds=inds)
    alt_onehot = encode_sequences(df['alt_sequence'], seqlen=seqlen, inds=inds)
    self.get_trained_model()

    if len(self.layers)==0:
      m = self.model
      ref_p = m.predict(ref_onehot)
      alt_p = m.predict(alt_onehot)

    else:
      print('Getting preds for all layers', flush=True)
      ref_ps, alt_ps = [], []
      for l in self.layers:
        # if l == 5 and re.search('dq', self.experiment_name): # if self.model.layers[5].__class__ == 'Bidirectional'
        #   print('Trying pooled layer activations')
        #   ref_p = self.model_class.pooled_layer_activations(l, ref_onehot)
        #   alt_p = self.model_class.pooled_layer_activations(l, alt_oneho5)
        # else:
        print('Getting preds for layer {}'.format(l), flush=True)
        ref_p = self.model_class.layer_activations(l, ref_onehot)
        alt_p = self.model_class.layer_activations(l, alt_onehot)
        if len(ref_p.shape)==3:
          ref_p = np.mean(ref_p, axis=1)
          alt_p = np.mean(alt_p, axis=1)
        ref_ps.append(ref_p)
        alt_ps.append(alt_p)
      ref_p = np.concatenate(ref_ps, axis=1)
      alt_p = np.concatenate(alt_ps, axis=1)
    return ref_p, alt_p

  def get_features(self, df, elem=None):
    reffname = 'data/cagi5_mpra/{}_ref_preds.npy'.format(self.experiment_name + self.filesuffix)
    train_inds = df.index.values
    if os.path.isfile(reffname):

      print('loading saved preds', reffname)
      ref_p = np.load(reffname)[train_inds]
      alt_p = np.load(reffname.replace('ref', 'alt'))[train_inds]

      if self.alllayers:
        self.model_class = self.get_untrained_model()
        all_sizes = [self.model_class.model.layers[l].output_shape[-1] for l in self.layers]
        assert np.sum(all_sizes) == ref_p.shape[1]

        endpoints = np.cumsum(all_sizes)
        ref_ps, alt_ps = [], []
        for l in self.layers:
          ind = all_layers.index(l)
          endpoint = endpoints[ind]
          start = endpoint - all_sizes[ind]
          print(l, start, endpoint)
          ref_ps.append(ref_p[:, start:endpoint])
          alt_ps.append(alt_p[:, start:endpoint])
        ref_p = np.concatenate(ref_ps, axis=1)
        alt_p = np.concatenate(alt_ps, axis=1)
    else:
      print('calculating preds')
      ref_p, alt_p = self.get_refalt_preds(df)
    return snp_feats_from_preds(ref_p, alt_p, self.feattypes)

  def get_trained_model(self):
    if not self.layers:
      from keras.models import load_model
      m = load_model('/home/arh96/consworkdir/results/models-best/models-best/{}.h5'.format(self.experiment_name))
      print('model loaded', flush=True)
      self.model = m
    else:
      model_class = self.get_untrained_model()
      model_class.get_compiled_model()
      model_class.model.load_weights('/home/arh96/consworkdir/results/models-best/models-best/{}.h5'.format(self.experiment_name))
      print('weights loaded', flush=True)
      self.model_class = model_class
      self.model = self.model_class.model

  def get_untrained_model(self):
    settings = pickle.load(open('data/remote_workspace/experiment_settings/{}.p'.format(self.experiment_name), 'rb'))
    # if i just want the class name i can do settings['model_class'].__name__
    model_class, model_args = settings['model_class'], settings['model_args'] 
    return model_class(**model_args)


class SNPContext(Features):
  # idea here is to use some kind of local information
  def __init__(self, context_size=2, raw_aggs=['max', 'mean', 'median'], abs_aggs=[]):
    # maybe also enable specification of what kinds of aggregate to perform
    self.context_size = context_size
    self.right_context_size = context_size // 2
    self.left_context_size = context_size - self.right_context_size
    self.raw_aggs = raw_aggs

  def get_features(self, df, elem=None):
    breakpoint_df = get_breakpoint_df(df)
    breakpoint_df['is_start'] = breakpoint_df['is_break'] == 'start'
    breakpoint_df['chunk_id'] = breakpoint_df.groupby(['regulatory_element'])['is_start'].cumsum() - 1
    breakpoint_df['AbsValue'] = np.abs(breakpoint_df['Value'])
    grouped_agg = breakpoint_df.groupby(['regulatory_element', 'chunk_id'])['Value'].agg(self.raw_aggs)

    context_features = {}
    nfeat = len(grouped_agg.keys())
    for regulatory_element, chunk_id in grouped_agg.index:
      max_chunk_id = grouped_agg.loc[regulatory_element].index.max()
      contfeat = []
      for i in range(self.left_context_size):
        ch = chunk_id - (i+1)
        assert ch != chunk_id
        if ch >= 0:
          contfeat.append(grouped_agg.loc[(regulatory_element, ch)].values)
        else:
          contfeat.append(np.zeros(nfeat))
      for i in range(self.right_context_size):
        ch = chunk_id + (i+1)
        assert ch != chunk_id
        if ch <= max_chunk_id:
          contfeat.append(grouped_agg.loc[(regulatory_element, ch)].values)
        else:
          contfeat.append(np.zeros(nfeat))        

      contfeat = np.concatenate(contfeat)
      context_features[(regulatory_element, chunk_id)] = contfeat

    featmat = np.zeros((len(df), nfeat*self.context_size))
    for i, (ix, row) in enumerate(breakpoint_df.iterrows()):
      featmat[i,:] = context_features[(row['regulatory_element'], row['chunk_id'])]
    return featmat


class Conservation(Features):
  def __init__(self, scores=['phastCon', 'phyloP', 'GerpN', 'GerpRS']):
    self.scores = scores

  def get_features(self, df, elem=None):
    feat = df[self.scores]
    return feat


class EnhancerOneHot(Features):
  def __init__(self, enh_names=['release_F9', 'release_GP1BB', 'release_HBB', 'release_HBG1',
       'release_HNF4A', 'release_IRF4', 'release_IRF6', 'release_LDLR',
       'release_MSMB', 'release_MYCrs6983267', 'release_PKLR',
       'release_SORT1', 'release_TERT-GBM', 'release_TERT-HEK293T',
       'release_ZFAND3']):
    self.enh_names = enh_names

  def get_features(self, df, elem=None):
    # https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present
    enhancers = df['regulatory_element'].astype(CategoricalDtype(categories=self.enh_names))
    onehot = pd.get_dummies(enhancers).values
    # print(onehot.shape)
    # other features: enhancer mean, enhancer same substitution
    return onehot


class SubstitutionOneHot(Features):
  def __init__(self):
    self.base_names = ['A', 'C', 'G', 'T']

  def get_features(self, df, elem=None):
    # https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present
    refs = df['Ref'].astype(CategoricalDtype(categories=self.base_names))
    alts = df['Alt'].astype(CategoricalDtype(categories=self.base_names))
    return np.concatenate([pd.get_dummies(refs).values, pd.get_dummies(alts).values], axis=1)


class MPRATransfer(Features):
  pass


class MultiFeatures(Features):
  """
  Concatenate several features together.
  """
  def __init__(self, features):
    self.features = features

  def get_features(self, df, elem=None):
    return np.concatenate([f.get_features(df, elem) for f in self.features], axis=1)
