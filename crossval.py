import re
import numpy as np
import pandas as pd

from cagi5_utils import get_breakpoint_df, get_chunk_counts
from models import *


class CVOperator:
  """
  Fits a model to a training data set and makes predictions on a
  validation data set.
  """

  def __init__(self, model_class, model_args=[], model_kwargs={}):
    self.model = model_class(*model_args, **model_kwargs)


  def get_preds(self, train_df, val_df, elem=None):
    """
    The model cannot supply stacked features so we add an additional
    argument here to supply them.
    """
    X_train = self.model.get_features(train_df, elem)
    y_train = self.model.get_response(train_df)

    self.model.fit(X_train, y_train)

    return self.make_predictions(val_df, elem)


  def make_predictions(self, df, elem=None):
    X = self.model.get_features(df, elem)
    return self.model.predict(X, index=df.index)



class PerElementOperator:
  """
  Groups data by regulatory element, applies a CVOperator
  to each group and combines predictions on validation
  data.
  """

  def __init__(self, child):
    self.child = child  # The child CVOperator


  def get_preds(self, train_df, val_df):
    train_grpd = train_df.groupby('regulatory_element')
    val_grpd = val_df.groupby('regulatory_element')
    preds = []
    for group, elem_train_df in train_grpd:
      try:
        elem_val_df = val_grpd.get_group(group)
        preds.append(self.child.get_preds(elem_train_df, elem_val_df))
      except KeyError:  # We don't have validation data for every group
        pass
    return pd.concat(preds)


# N.B. the way things are setup now, the deepsea stuff provides an
# independent model baseline, according to which each eqtl is modelled independently
# i.e. we learn f(snp) = y, where f is a function of features of a single snp, 
# and knows nothing (explicit) about enhancer or local context

class ChunkCV:
  """
  Break the data into cross-validation chunks and predict on held-out
  validation folds.
  """

  def __init__(self,
               df,
               operator,
               nf=5,
               fold_dict=None):
    self.operator = operator
    self.breakpoint_df = get_breakpoint_df(df)
    assert np.sum(self.breakpoint_df['chunk_length']) == sum(df.groupby(['regulatory_element'])['Pos'].nunique())
    self.nf = nf
    if fold_dict is None:
      fold_dict = df_cv_split(self.breakpoint_df, nf)
    self.fold_dict = fold_dict


  def get_cv_preds(self):
    for f in range(self.nf):
      print('Getting predictions for fold {}'.format(f))
      self.get_fold_preds(f)
    return self.breakpoint_df


  def get_fold_preds(self, f):
    trainvaldf = train_val_split(self.fold_dict, self.breakpoint_df, f)
    train_df = trainvaldf[trainvaldf['is_train']]
    val_df = trainvaldf[~trainvaldf['is_train']]
    preds = self.operator.get_preds(train_df, val_df)
    for col in preds.columns.values:
      self.breakpoint_df.loc[~self.breakpoint_df['is_train'], col] = preds[col]
    return preds


def df_cv_split(breakpoint_df, nf):
  chunk_counts = get_chunk_counts(breakpoint_df)
  fold_dict = pick_train_chunks(chunk_counts, nf=nf)
  return fold_dict


def pick_train_chunks(chunk_count_df, nf=5):
  """
  Creates a length nf (num folds) list for each element of which chunks to include in each cross val fold.
  Chunks are randomly selected for each fold.
  """
  fold_dict = {}
  for reg_el, row in chunk_count_df.iterrows():
    chunk_ids = np.arange(int(row['count']))
    np.random.shuffle(chunk_ids)
    chunk_ids = list(chunk_ids)
    leftovers = int(row['count']) % nf
    base_chunks_per_fold = np.int(row['count'] // nf)
    fold_chunk_ids = [chunk_ids[i*base_chunks_per_fold:(i+1)*base_chunks_per_fold] for i in range(nf)]
    for l in range(leftovers):
      fold_chunk_ids[l].append(chunk_ids[-(l+1)])
    fold_dict[reg_el] = fold_chunk_ids

  return fold_dict


def train_val_split(fold_dict, df, fold=0):
  df['chunk_id'] = df['chunk_id'].astype(int)
  df['is_train'] = True
  for reg_el, fold_chunks in fold_dict.items():
    val_chunks = fold_chunks[fold]
    df.loc[(df['regulatory_element']==reg_el)&(df['chunk_id'].isin(val_chunks)), 'is_train'] = False
  return df
