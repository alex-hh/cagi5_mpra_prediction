import pickle
import numpy as np
import xgboost as xgb

from utils import snp_feats_from_preds, encode_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


class DeepSeaSNP:
  
  def __init__(self, use_saved_preds=True, feattypes=['absdiff'],
               verbose=False, multiclass='ovr', classifier='lr', classifier_kwargs={}):
    self.use_saved_preds = use_saved_preds
    self.feattypes = feattypes
    self.verbose = verbose
    self.multiclass = multiclass
    self.classifier_kwargs = classifier_kwargs
    self.classifier = classifier

  def get_features(self, df):
    if self.use_saved_preds:
      train_inds = df.index.values
      train_ref = np.load('data/cagi5_mpra/deepsea_ref_preds.npy')[train_inds]
      train_alt = np.load('data/cagi5_mpra/deepsea_alt_preds.npy')[train_inds]

    return snp_feats_from_preds(train_ref, train_alt, self.feattypes)

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


class DSDataKerasModel:
  def __init__(self, experiment_name, feattypes=['absdiff'],
               verbose=False, multiclass='ovr', classifier='lr', classifier_kwargs={}):
    self.feattypes = feattypes
    self.verbose = verbose
    self.multiclass = multiclass
    self.classifier_kwargs = classifier_kwargs
    self.classifier = classifier
    self.experiment_name = experiment_name

  def get_features(self, df):
    assert 'ref_sequence' in df.columns
    ref_onehot = encode_sequences(df['ref_sequence'], seqlen=1000)
    alt_onehot = encode_sequences(df['alt_sequence'], seqlen=1000)
    self.get_trained_model()

    if all_layers:
      m = model_class.model
      ref_p = m.predict(ref_onehot)
      alt_p = m.predict(alt_onehot)
      pass

    else:
      pass

  def get_trained_model(self):
    model_class = self.get_untrained_model(self.experiment_name)
    model_class.get_compiled_model()
    model_class.model.load_weights('data/remote_results/models-best/{}.h5'.format(RESULT_DIR, experiment_name))
    self.model_class = model_class

  def get_untrained_model(self):
    settings = pickle.load(open('data/remote_workspace/experiment_settings/{}.p'.format(self.experiment_name), 'rb'))
    # if i just want the class name i can do settings['model_class'].__name__
    model_class, model_args = settings['model_class'], settings['model_args'] 
    if 'reg_weighting' in model_args: 
      model_args['reg_weighting'] = get_weighting(**model_args['reg_weighting'])
    return model_class(**model_args)