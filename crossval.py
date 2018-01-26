import re
import numpy as np
import pandas as pd

from cagi5_utils import get_breakpoint_df, get_chunk_counts
from models import *

class CVOperator:

  def __init__(self, df, model_class, model_args=[], model_kwargs={}):
    self.model = model_class(*model_args, **model_kwargs)
    self.df = df
    self.df['cv_prediction'] = np.nan

  def get_fold_preds(self, train_df, val_df):
    X_train = self.model.get_features(train_df)
    y_train = train_df['class']

    self.model.fit(X_train, y_train)

    X_val = self.model.get_features(val_df)
    y_val = val_df['class']

    preds = self.model.predict(X_val)
    return preds

def cvpreds_df_enhancer_folds(df, model_class, model_args=[], model_kwargs={}):
  model = model_class(*model_args, **model_kwargs)
  df['cv_prediction'] = np.nan
  if 'base_element' not in df.columns:
    df['base_element'] = df.apply(lambda row: row['regulatory_element'][8:], axis=1)
    df['base_element'] = df.apply(lambda row: 'TERT' if re.match('TERT', row['base_element'])\
                                  else row['base_element'], axis=1)
  for val_element in df['base_element'].unique():
    train_df = df[df['base_element']!=val_element]
    val_df = df[df['base_element']==val_element]
    train_inds = train_df.index.values
    val_inds = val_df.index.values

    X_train = model.get_features(train_df)
    y_train = train_df['class']

    model.fit(X_train, y_train)

    X_val = model.get_features(val_df)
    y_val = val_df['class']

    preds = model.predict(X_val)

    df.loc[df['base_element']==val_element, 'cv_prediction'] = preds

  return df

# N.B. the way things are setup now, the deepsea stuff provides an
# independent model baseline, according to which each eqtl is modelled independently
# i.e. we learn f(snp) = y, where f is a function of features of a single snp, 
# and knows nothing (explicit) about enhancer or local context

class ChunkCV(CVOperator):

  def __init__(self, df, model_class, model_args=[], model_kwargs={}, nf=5):
    super().__init__(df, model_class, model_args=model_args, model_kwargs=model_kwargs)
    self.breakpoint_df = get_breakpoint_df(df)
    self.breakpoint_df['is_start'] = self.breakpoint_df['is_break'] == 'start'
    self.breakpoint_df['chunk_id'] = self.breakpoint_df.groupby(['regulatory_element'])['is_start'].cumsum() - 1
    assert np.sum(self.breakpoint_df['chunk_length']) == sum(df.groupby(['regulatory_element'])['Pos'].nunique())
    self.nf = nf
    self.chunk_counts = get_chunk_counts(self.breakpoint_df)
    self.fold_dict = pick_train_chunks(self.chunk_counts, nf=self.nf)

  def get_cv_preds(self):
    for f in range(self.nf):
      trainvaldf = train_val_split(self.fold_dict, self.breakpoint_df, f)
      train_df = trainvaldf[trainvaldf['is_train']]
      val_df = trainvaldf[~trainvaldf['is_train']]
      preds = self.get_fold_preds(train_df, val_df)
      self.breakpoint_df.loc[~self.breakpoint_df['is_train'], 'cv_prediction'] = preds
    return self.breakpoint_df


def cvpreds_df_chunk_folds(df, model_class, model_args=[], model_kwargs={}, nf=5):
  model = model_class(*model_args, **model_kwargs)
  df['cv_prediction'] = np.nan

  breakpoint_df = get_breakpoint_df(df)
  breakpoint_df['is_start'] = breakpoint_df['is_break'] == 'start'
  breakpoint_df['chunk_id'] = breakpoint_df.groupby(['regulatory_element'])['is_start'].cumsum() - 1

  assert np.sum(breakpoint_df['chunk_length']) == sum(df.groupby(['regulatory_element'])['Pos'].nunique())
  chunk_counts = get_chunk_counts(breakpoint_df)
  fold_dict = pick_train_chunks(chunk_counts, nf=nf)
  
  for f in range(nf):
    trainvaldf = train_val_split(fold_dict, breakpoint_df, f)
    train_df = trainvaldf[trainvaldf['is_train']]
    val_df = trainvaldf[~trainvaldf['is_train']]

    X_train = model.get_features(train_df)
    y_train = train_df['class']

    model.fit(X_train, y_train)

    X_val = model.get_features(val_df)
    y_val = val_df['class']

    preds = model.predict(X_val)

    breakpoint_df.loc[~breakpoint_df['is_train'], 'cv_prediction'] = preds

  return breakpoint_df



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

