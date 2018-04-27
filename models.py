import os
import multiprocessing
import pickle
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import xgboost as xgb
import lightgbm as lgbm
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
    else:
      return ValueError('Unknown model name: {}'.format(self.model_name))
    sample_weight = compute_sample_weight('balanced', y) # not sure if classes need to be labelled 0,1,2 (if so can use label encoder)
    self.model_value.fit(X, y['Value'], sample_weight=sample_weight)
    self.model_conf.fit(X, y['Confidence'], sample_weight=sample_weight)
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


class Features(ABC):
  """
  Abstract base class for features.
  """

  @abstractmethod
  def get_features(self, df, elem=None):
    pass


class DeepSeaSNP(Features):

  def __init__(self, use_saved_preds=True, feattypes=['diff']):
    self.use_saved_preds = use_saved_preds
    self.feattypes = feattypes

  def get_features(self, df, elem=None):
    if self.use_saved_preds:
      train_inds = df.index.values
      train_ref = np.load('data/cagi5_mpra/deepsea_ref_preds.npy')[train_inds]
      train_alt = np.load('data/cagi5_mpra/deepsea_alt_preds.npy')[train_inds]

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

  def __init__(self, idxs = None):
    self.feats = np.load('data/dnase-features.npy')
    if idxs is not None:
      self.feats = self.feats[:, idxs]

  def get_features(self, df, elem=None):
    return self.feats[df.index.values]


class DSDataKerasModel(Features):

  def __init__(self, experiment_name, feattypes=['diff'], alllayers=False, layers=[]):
    self.feattypes = feattypes
    self.experiment_name = experiment_name
    self.layers = layers
    self.alllayers = alllayers

  def get_refalt_preds(self, df):
    assert 'ref_sequence' in df.columns
    ref_onehot = encode_sequences(df['ref_sequence'], seqlen=1000)
    alt_onehot = encode_sequences(df['alt_sequence'], seqlen=1000)
    self.get_trained_model()

    if len(self.layers)==0:
      m = self.model_class.model
      ref_p = m.predict(ref_onehot)
      alt_p = m.predict(alt_onehot)

    else:
      ref_ps, alt_ps = [], []
      for l in self.layers:
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
    suffix = ''
    if self.alllayers:
      suffix = '-all'
    reffname = 'data/cagi5_mpra/{}_ref_preds.npy'.format(self.experiment_name + suffix)
    train_inds = df.index.values
    if os.path.isfile(reffname):

      print('loading saved preds', reffname)
      ref_p = np.load(reffname)[train_inds]
      alt_p = np.load(reffname.replace('ref', 'alt'))[train_inds]

      self.model_class = self.get_untrained_model()
      all_layers = [3,5,11]
      all_sizes = [self.model_class.model.layers[l].output_shape[-1] for l in all_layers]
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
    model_class = self.get_untrained_model()
    model_class.get_compiled_model()
    model_class.model.load_weights('data/remote_results/models-best/{}.h5'.format(self.experiment_name))
    self.model_class = model_class

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
  def __init__(self, features):
    self.features = features

  def get_features(self, df, elem=None):
    return np.concatenate([f.get_features(df, elem) for f in self.features], axis=1)
