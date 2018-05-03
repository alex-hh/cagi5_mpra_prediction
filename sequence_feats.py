from utils import encode_sequences, get_sequences
from models import DSDataKerasModel


def seqfeats_from_df(df, seqlen=None, seqfeatextractor='deepsea',
                     use_gpu=False, layers=[]):
  from deepsea import DeepSea
  if 'ref_sequence' not in df.columns:
    print('getting sequences')
    ref_sequences, alt_sequences = get_sequences(df, which_set='cagi4')
  else:
    ref_sequences = df['ref_sequence']
    alt_sequences = df['alt_sequence']
    snp_inds = df['snp_index']
    print(snp_inds, flush=True)
  ref_onehot = encode_sequences(ref_sequences, seqlen=seqlen, inds=snp_inds)
  alt_onehot = encode_sequences(alt_sequences, seqlen=seqlen, inds=snp_inds)
  if seqfeatextractor == 'deepsea':
    # if all_layers:
    #   features = ['2','6','9','13','15']
    #   ds = DeepSea(use_gpu=use_gpu, features=features)
    #   ref_preds = ds.layer_activations(ref_onehot)
    #   alt_preds = ds.layer_activations(alt_onehot)
    # else:
    #   features = ['15']

      ds = DeepSea(use_gpu=use_gpu, features=layers)
      ref_preds = ds.predict(ref_onehot)
      alt_preds = ds.predict(alt_onehot)
  elif seqfeatextractor == 'danqpy2':
    pass
  else:
    feat_extractor = DSDataKerasModel(experiment_name=seqfeatextractor,
                                      layers=layers)
    ref_preds, alt_preds = feat_extractor.get_refalt_preds(df, seqlen=seqlen)
  return ref_preds, alt_preds