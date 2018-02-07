import re
import sys

import pandas as pd

from sequence_feats import seqfeats_from_df
from utils import get_seqs_and_inds
# absdiff = snpfeats_from_df(df, 1000)

def main(seqfeatextractor, alllayers=0):
  df = pd.read_csv('data/cagi5_df.csv')
  refs, alts, inds = get_seqs_and_inds(df)
  df['ref_sequence'] = refs
  df['alt_sequence'] = alts
  df['snp_index'] = inds
  df['Refcheck'] = df.apply(lambda row: row['ref_sequence'][row['snp_index']], axis=1)
  df['Altcheck'] = df.apply(lambda row: row['alt_sequence'][row['snp_index']], axis=1)

  assert list(df['Refcheck']) == list(df['Ref'])
  assert list(df['Altcheck']) == list(df['Alt'])

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
      layers = [11]

  print('getting preds', flush=True)
  refp, altp = seqfeats_from_df(df, use_gpu=False, seqlen=1000,
                                layers=layers)
  print('ref preds shape', refp.shape, flush=True)

  suffix = ''
  if alllayers:
    suffix = '-all'

  np.save('data/cagi5_mpra/{}_ref_preds.npy'.format(seqfeatextractor + suffix), refp)
  np.save('data/cagi5_mpra/{}_alt_preds.npy'.format(seqfeatextractor + suffix), altp)

if __name__ == '__main__':
  seqfeatextractor = sys.argv[1]
  if len(sys.argv) > 2:
    alllayers = int(sys.argv[2]) # 0 or 1
  main(seqfeatextractor, alllayers=alllayers)