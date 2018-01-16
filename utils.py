import pysam
import numpy as np

from constants import LOCS
from deepsea import DeepSea


def get_sequences_cagi5(df):
  fasta_file = 'data/remote_data/hg19.genome.fa'
  ref_sequences = []
  alt_sequences = []
  with pysam.Fastafile(fasta_file) as genome:
    for i, (ix, row) in enumerate(df.iterrows()):
      reg_el_code = row['regulatory_element'][8:]
      seqstart = LOCS[reg_el_code]['start']
      seqend = LOCS[reg_el_code]['end']
      rel_pos = pos-f9_start-1
      dnastr = genome.fetch('chr'+row['#Chrom'], start, end).upper()
      assert dnastr[rel_pos] == row['Ref']
      print(dnastr[rel_pos], row['Ref'])

def get_sequences(df, which_set='cagi4'):
  """
    cagi4: extract 150bp from hg19 centred on the variant.
    For each variant, a pair of 150-nt candidate regulatory sequence oligonucleotides was synthesized
    with the variant located at the central position (i.e. SNP at position 76).
    For insertion-deletion variants, the longer of the two alleles was designed as a 150-nt oligonucleotide; 
    the shorter allele was then designed with the same flanking sequences as the longer allele 
    (e.g., for a single-nucleotide InDel TC/C: 74N[TC]74N and 74N[T]74N

    cagi5: then extract an amount depending on the specific promoter / enhancer in question.
  """
  fasta_file = 'data/remote_data/hg19.genome.fa'
  ref_sequences = []
  alt_sequences = []
  df['sequence'] = 'NA'
  if which_set == 'cagi4':
    with pysam.Fastafile(fasta_file) as genome:
      for i, (ix, row) in enumerate(df.iterrows()):
        if len(row['RefAllele']) >= len(row['AltAllele']):
          seqstart = int(row['pos']) - (76 - (len((row['RefAllele']))//2)) # to centre things this should work
          dnastr = genome.fetch('chr' + str(row['chr']), seqstart, seqstart + 150).upper()
          ref_start = (75 - (len((row['RefAllele']))//2))
          ref_end = 76+(len((row['RefAllele']))-1)//2

          alt = list(str(dnastr))
          alt = ''.join(alt[:ref_start] + list(row['AltAllele']) + alt[ref_end:])
          
          assert len(dnastr) == 150

        else:

          seqstart = int(row['pos']) - (76 - (len((row['AltAllele']))//2)) # to centre things this should work
          seqend = seqstart + 150 - (len(row['AltAllele']) - len(row['RefAllele']))
          dnastr = genome.fetch('chr' + str(row['chr']), seqstart, seqend).upper()

          ref_start = row['pos'] - seqstart - 1 
          ref_end = ref_start + len(row['RefAllele'])

          alt = list(str(dnastr))
          alt = ''.join(alt[:ref_start] + list(row['AltAllele']) + alt[ref_end:])

          assert len(alt) == 150

        assert alt[ref_start:ref_start + len(row['AltAllele'])] == row['AltAllele']
        assert dnastr[ref_start: ref_end] == row['RefAllele']
        ref_sequences.append(dnastr)
        alt_sequences.append(alt)

  elif which_set == 'cagi5':
    dnastr = genome_open.fetch(row['chr'], enhancer_start, enhancer_end).upper()

  return ref_sequences, alt_sequences

def encode_strings(dnastrs, dims=['A', 'G', 'C', 'T']):
  """
  dnastrs: a list of dnastrings, all of the same length, or a single string
  """
  if type(dnastrs) == str:
    arr = np.zeros((len(dnastrs),4))
    for j, c in enumerate(dnastrs):
      if c in dims:
        arr[j, dims.index(c)] = 1
  else:
    seqlen = len(dnastrs[0])
    arr = np.zeros((len(dnastrs), seqlen, 4))   
    for i, dnastr in enumerate(dnastrs):
        assert len(dnastr) == seqlen
        for j, c in enumerate(dnastr):
          if c in dims:
            arr[i, j, dims.index(c)] = 1
  return arr

def snp_feats_from_preds(ref_preds, alt_preds, feattypes=[]):
  # todo: could multiply the difference by the max of the preds 'scalediff': a difference to an 'on' feature is more significant than a difference to an off feature
  calculated_feats = []
  if 'absdiff' in feattypes:
    abs_diff_feats = np.abs(ref_preds-alt_preds)
    calculated_feats.append(abs_diff_feats)
  if 'diff' in feattypes:
    calculated_feats.append(ref_preds-alt_preds)
  if 'scaleddiff' in feattypes:
    calculated_feats.append(np.abs((ref_preds-alt_preds)*np.max(np.stack((ref_preds, alt_preds), axis=-1), axis=2)))
  if 'absodds' in feattypes or 'odds' in feattypes:
    clipped_ref = np.clip(ref_preds, 1e-7, 1-1e-7)
    clipped_alt = np.clip(alt_preds, 1e-7, 1-1e-7)
    odds_ref = clipped_ref/(1-clipped_ref)
    odds_alt = clipped_alt/(1-clipped_alt)
    log_odds_ratio = np.log2(odds_ref/odds_alt)
    if 'absodds' in feattypes:
      calculated_feats.append(np.abs(log_odds_ratio))
    if 'odds' in feattypes:
      calculated_feats.append(log_odds_ratio)
  return np.concatenate(calculated_feats, axis=1)

def encode_sequences(sequences, seqlen=None):
  # N.B. that the fact that Basenji, for example, does binned predictions should mean that
  # it actually can be applied to variable length sequences
  preprocessed_seqs = []
  if seqlen is None:
    # just encode without padding
    preprocessed_seqs = sequences
  else:
    for seq in sequences:
      assert seqlen > len(seq)
      pad_left = (seqlen - len(seq))//2
      pad_right = seqlen - (len(seq) + pad_left)
      seq = 'N'*pad_left + seq + 'N'*pad_right
      assert len(seq) == seqlen
      preprocessed_seqs.append(seq)

  return encode_strings(preprocessed_seqs)

def seqfeats_from_df(df, seqlen=None, seqfeatextractor='deepsea',
                     use_gpu=False, all_layers=False):
  if 'ref_sequence' not in df.columns:
    print('getting sequences')
    ref_sequences, alt_sequences = get_sequences(df, which_set='cagi4')
  else:
    ref_sequences = df['ref_sequence']
    alt_sequences = df['alt_sequence']
  ref_onehot = encode_sequences(ref_sequences, seqlen=seqlen)
  alt_onehot = encode_sequences(alt_sequences, seqlen=seqlen)
  if seqfeatextractor == 'deepsea':
    if all_layers:
      features = ['2','6','9','13','15']
      ds = DeepSea(use_gpu=use_gpu, features=features)
      ref_preds = ds.layer_activations(ref_onehot)
      alt_preds = ds.layer_activations(alt_onehot)
    else:
      features = ['15']
      ds = DeepSea(use_gpu=use_gpu, features=features)
      ref_preds = ds.predict(ref_onehot)
      alt_preds = ds.predict(alt_onehot)
  return ref_preds, alt_preds

def snpfeats_from_df(df, seqlen=None, seqfeatextractor='deepsea',
                     compfeattype='absdiff',
                     use_gpu=False):
  ref_preds, alt_preds = seqfeats_from_df(df, seqlen=seqlen,
                                          seqfeatextractor=seqfeatextractor,
                                          use_gpu=use_gpu)
  feats = snp_feats_from_preds(ref_preds, alt_preds,
                               feattypes=[compfeattype] if type(compfeattype)==str else compfeattype)
  return feats

def score_preds(preds, df=None, y=None):
  if df is not None:
    y = df['emVar_Hit']
  else:
    assert y is not None
  roc_score = roc_auc_score(y, preds)
  auprc_score = average_precision_score(y, preds)
  print('AUROC:\t{}\tAUPRC:\t{}'.format(roc_score, auprc_score))
  return roc_score, auprc_score
