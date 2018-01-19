import re
import numpy as np

from models import *


def cvpreds_df(df, model_class, model_args=[], model_kwargs={}):
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