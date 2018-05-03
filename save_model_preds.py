import re
import sys
import argparse

import pandas as pd
import numpy as np

from sequence_feats import seqfeats_from_df
from utils import get_seqs_and_inds
# absdiff = snpfeats_from_df(df, 1000)

def main(seqfeatextractor, layer=None, alllayers=False, dataset='train', seqlen=1000, v=2):
  if layer > 0:
    assert not alllayers
    layers = [layer]
  else:
    if seqfeatextractor == 'deepsea' and alllayers:
      layers = ['2','6','9','13','15']
    elif seqfeatextractor == 'deepsea':
      layers = ['15']
    elif seqfeatextractor == 'danqpy2':
      pass
    elif re.search('dq', seqfeatextractor):
      if alllayers:
        layers = [3,5,11]
      else:
        layers = []
    else:
      layers = []

  print('Saving preds from {} model layers {} on {} set using seqs of len {} seqs v {}'.format(
    seqfeatextractor, '-'.join([str(l) for l in layers]), dataset, seqlen, v))
  if dataset == 'train':
    df = pd.read_csv('data/cagi5_df.csv')
  elif dataset == 'test':
    df = pd.read_csv('data/cagi5_mpra/TestDataset.txt', sep='\t')
  refs, alts, inds = get_seqs_and_inds(df, v=v)
  df['ref_sequence'] = refs
  df['alt_sequence'] = alts
  df['snp_index'] = inds
  df['Refcheck'] = df.apply(lambda row: row['ref_sequence'][row['snp_index']], axis=1)
  df['Altcheck'] = df.apply(lambda row: row['alt_sequence'][row['snp_index']], axis=1)

  assert list(df['Refcheck']) == list(df['Ref'])
  assert list(df['Altcheck']) == list(df['Alt'])

  

  print('getting preds', flush=True)
  refp, altp = seqfeats_from_df(df, use_gpu=False, seqlen=seqlen, # target sequence length - i.e. how much padding should be applied
                                layers=layers, seqfeatextractor=seqfeatextractor)
  print('ref preds shape', refp.shape, flush=True)

  suffix = ''
  if layers:
    suffix = '-'+'-'.join([str(l) for l in layers])
  if dataset == 'test':
    suffix += '-test'

  np.save('data/cagi5_mpra/{}_ref_preds_v1.npy'.format(seqfeatextractor + suffix), refp)
  np.save('data/cagi5_mpra/{}_alt_preds_v1.npy'.format(seqfeatextractor + suffix), altp)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("seqfeatextractor", help="Name of model to use to generate preds")
  parser.add_argument("--test", action='store_true')
  parser.add_argument("seqlen", nargs='?', default=1000, type=int)
  parser.add_argument("--layer", default=0, type=int)
  parser.add_argument("--v", default=2, type=int)
  parser.add_argument('--alllayers', action='store_true')
  args = parser.parse_args()
  dataset = 'train'
  if args.test:
    dataset = 'test'
  main(args.seqfeatextractor, layer=args.layer, seqlen=args.seqlen, alllayers=args.alllayers,
       dataset=dataset, v=args.v)