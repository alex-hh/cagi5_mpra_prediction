import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, \
        precision_recall_curve, auc
from sklearn.preprocessing.label import LabelBinarizer
from models import Classifier, Regression
from models import DeepSeaSNP, Conservation, SNPContext, MultiFeatures, EnhancerOneHot, \
        SubstitutionOneHot, DNase, DSDataKerasModel
from cagi5_utils import get_breakpoint_df
from crossval import ChunkCV, CVOperator, df_cv_split
from utils import make_plots

df = pd.read_csv('data/cagi5_df.csv')

breakpoint_df = get_breakpoint_df(df)

nfolds = 5
fold_dict = df_cv_split(breakpoint_df, nfolds)

dskerasfeats = DSDataKerasModel('crnn_500_200', feattypes=['diff'], layers=[], alllayers=False)
deepseadiffs = DeepSeaSNP(feattypes=['diff'])

flist = [dskerasfeats, deepseadiffs]

for feats in flist:

  cv_chunk = ChunkCV(df,
                   operator=CVOperator(
                                      Classifier,
                                      model_kwargs={'features': feats, 'model_name': 'lr'}),
                    fold_dict=fold_dict)
  cvdf_chunk = cv_chunk.get_cv_preds()

  binarizer = LabelBinarizer()
  ybin = binarizer.fit_transform(cvdf_chunk['class'])
  print(binarizer.classes_)
  print(roc_auc_score(ybin, binarizer.transform(cvdf_chunk['PredClass']), average=None))
