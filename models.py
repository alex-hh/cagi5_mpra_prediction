import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from utils import snp_feats_from_preds, encode_sequences
from cagi5_utils import get_breakpoint_df, get_chunk_counts

class BaseModel:

  def __init__(self, classifier='lr',
               multiclass='ovr',
               classifier_kwargs={}, verbose=False):
    self.classifier_kwargs = classifier_kwargs
    self.classifier = classifier
    self.multiclass=multiclass
    self.verbose = verbose

  def fit(self, X, y):
    # In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    # scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    # entropy loss if the 'multi_class' option is set to 'multinomial'
    if self.classifier == 'lr':
      self.lr = LogisticRegression(penalty='l2', C=0.01, multi_class=self.multiclass)
    elif self.classifier == 'xgb':
      self.lr = xgb.XGBClassifier(**self.classifier_kwargs)
    sample_weight = compute_sample_weight('balanced', y) # not sure if classes need to be labelled 0,1,2 (if so can use label encoder)
    self.lr.fit(X, y, sample_weight=sample_weight)
    if self.verbose:
      print('Train accuracy: {}'.format(self.lr.score(X, y)))

  def predict(self, X):
    return self.lr.predict(X)

class DeepSeaSNP(BaseModel):
  
  def __init__(self, use_saved_preds=True, feattypes=['absdiff'],
               verbose=False, multiclass='ovr', classifier='lr', classifier_kwargs={}):
    self.use_saved_preds = use_saved_preds
    self.feattypes = feattypes
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df):
    if self.use_saved_preds:
      train_inds = df.index.values
      train_ref = np.load('data/cagi5_mpra/deepsea_ref_preds.npy')[train_inds]
      train_alt = np.load('data/cagi5_mpra/deepsea_alt_preds.npy')[train_inds]

    return snp_feats_from_preds(train_ref, train_alt, self.feattypes)


class DSDataKerasModel(BaseModel):
  def __init__(self, experiment_name, feattypes=['absdiff'],
               verbose=False, multiclass='ovr', classifier='lr', classifier_kwargs={}):
    self.feattypes = feattypes
    self.experiment_name = experiment_name
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df, layers=[]):
    assert 'ref_sequence' in df.columns
    ref_onehot = encode_sequences(df['ref_sequence'], seqlen=1000)
    alt_onehot = encode_sequences(df['alt_sequence'], seqlen=1000)
    self.get_trained_model()

    if len(layers)==0:
      m = self.model_class.model
      ref_p = m.predict(ref_onehot)
      alt_p = m.predict(alt_onehot)

    else:
      ref_ps, alt_ps = [], []
      for l in layers:
        ref_p = self.model_class.layer_activations(l, ref_onehot)
        alt_p = self.model_class.layer_activations(l, alt_onehot)
        ref_ps.append(ref_p)
        alt_ps.append(alt_p)
      ref_p = np.concatenate(ref_ps, axis=1)
      alt_p = np.concatenate(alt_ps, axis=1)

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

class SNPContext(BaseModel):
  # idea here is to use some kind of local information
  def __init__(self, context_size=2, raw_aggs=['max', 'mean', 'median'], abs_aggs=[],
               multiclass='ovr', classifier='lr', classifier_kwargs={}, verbose=False):
    # maybe also enable specification of what kinds of aggregate to perform
    self.context_size = context_size
    self.right_context_size = context_size // 2
    self.left_context_size = context_size - self.right_context_size
    self.raw_aggs = raw_aggs
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df):
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

class Conservation(BaseModel):
  def __init__(self, scores=['phastCon', 'phyloP', 'GerpN', 'GerpRS'],
               multiclass='ovr', classifier='lr', classifier_kwargs={},
               verbose=False):
    self.scores = scores
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df):
    feat = df[self.scores]
    return feat

class EnhancerOneHot(BaseModel):
  def __init__(self, multiclass='ovr', classifier='lr', classifier_kwargs={},
               verbose=False):
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df):
    onehot = pd.get_dummies(df['regulatory_element']).values
    return onehot

class MPRATransfer(BaseModel):
  pass

class MixedModel(BaseModel):
  def __init__(self, models=[], model_kwargs=[],
               multiclass='ovr', classifier='lr', classifier_kwargs={},
               verbose=False):
    print(models)
    self.models = [m(**kwargs) for m, kwargs in zip(models, model_kwargs)]
    super().__init__(multiclass=multiclass, classifier_kwargs=classifier_kwargs,
                     classifier=classifier, verbose=verbose)

  def get_features(self, df):
    features = [m.get_features(df) for m in self.models]
    features = np.concatenate(features, axis=1)
    return features