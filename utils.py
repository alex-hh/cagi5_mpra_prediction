import re
import csv
import pysam
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from constants import LOCS
from collections import defaultdict


def make_roc_plot(cvdf_chunk, col='PredClass'):
  """Makes a ROC plot of significant effect prediction."""
  fpr, tpr, thresholds = roc_curve(cvdf_chunk['class'].abs(), cvdf_chunk[col].abs(),
                                    pos_label=1)
  fig = plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  return fig


def compute_row_ref(row, base_seq_dict, use_modified=True):
  if use_modified:
    if row['regulatory_element']+'MOD' in base_seq_dict:
      ref_seq = base_seq_dict[row['regulatory_element']] + 'MOD'
    else:
      ref_seq = base_seq_dict[row['regulatory_element']]
  else:
    ref_seq = base_seq_dict[row['regulatory_element']]
  return ref_seq


def load_base_seqs(filepath='data/cagi5_mpra/base_seqs.csv'):
  base_seq_dict = dict()
  with open(filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
      reg_el_code, seq, seqstart = row
      base_seq_dict[reg_el_code] = {}
      base_seq_dict[reg_el_code]['seq'] = seq
      base_seq_dict[reg_el_code]['start'] = int(seqstart)
  return base_seq_dict

def get_seqs_and_inds(df, use_modified=True):
  refs = []
  alts = []
  inds = []
  base_seq_dict = load_base_seqs()
  for (ix, row) in df.iterrows():
    reg_el_code = row['regulatory_element'][8:]
    if re.match('TERT', reg_el_code):
      reg_el_code = 'TERT'
    if use_modified and reg_el_code+'MOD' in base_seq_dict:
      reg_el_code = reg_el_code + 'MOD'
    seq = base_seq_dict[reg_el_code]['seq']
    start = base_seq_dict[reg_el_code]['start']

    rel_pos = row['Pos'] - start - 1
    if use_modified:
      # print(row['regulatory_element'], rel_pos, reg_el_code, seq[rel_pos], row['Ref'])
      assert seq[rel_pos] == row['Ref']
    else:
      if seq[rel_pos] != row['Ref']:
        print('Non matching seq at index', ix)
    
    alt = list(seq)
    alt[rel_pos] = row['Alt']
    alt = ''.join(alt)

    refs.append(seq)
    alts.append(alt)
    inds.append(rel_pos)

  return refs, alts, inds

def save_base_seqs(df):
  fasta_file = 'data/remote_data/hg19.genome.fa'
  ref_sequences = set()
  ref_dict = {}
  ref, alt, inds, modified = get_check_sequences_cagi5(df)
  for (ix,row), ref, ind, mod in zip(df.iterrows(), ref, inds, modified):
    if ref not in ref_sequences:
      reg_el_code = row['regulatory_element'][8:]
      if re.match('TERT', reg_el_code):
        reg_el_code = 'TERT'
      if mod:
        reg_el_code = reg_el_code + 'MOD'
      ref_dict[reg_el_code] = ref
      ref_sequences.add(ref)

  with open('data/cagi5_mpra/base_seqs.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['regulatory_element', 'ref_sequence', 'start_pos'])
    for k, v in ref_dict.items():
      if re.search('MOD$', k):
        reg_el_code = k[:-3]
      else:
        reg_el_code = k
      seqstart = LOCS[reg_el_code]['start']
      writer.writerow([k, v, seqstart])

def get_check_sequences_cagi5(df):
  # n.b. we should actually only need to make n reg elements calls to the genome, and to do this only once.
  fasta_file = 'data/remote_data/hg19.genome.fa'
  ref_sequences = []
  alt_sequences = []
  snp_inds = []
  modified = []
  with pysam.Fastafile(fasta_file) as genome:
    for i, (ix, row) in enumerate(df.iterrows()):
      reg_el_code = row['regulatory_element'][8:]
      if re.match('TERT', reg_el_code):
        reg_el_code = 'TERT'
      seqstart = LOCS[reg_el_code]['start']
      seqend = LOCS[reg_el_code]['end']
      rel_pos = row['Pos']-seqstart-1
      try:
        assert rel_pos >= 0
      except AssertionError as e:
        seqstart = row['Pos'] - 1
        rel_pos = 0

      dnastr = str(genome.fetch('chr'+row['#Chrom'], seqstart, seqend).upper())
      alt = list(str(dnastr))
      alt[rel_pos] = row['Alt']
      alt = ''.join(alt)

      try:
        assert dnastr[rel_pos] == row['Ref'],\
        '{} does not match row ref {}, position {} chr {} in {} CRE'.format(dnastr[rel_pos], row['Ref'], 
                                    row['Pos'], row['#Chrom'], row['regulatory_element'])
        modified.append(False)
      except AssertionError as e:
        print(e)
        print(rel_pos)
        dnastr = list(dnastr)
        dnastr[rel_pos] = row['Ref']
        dnastr = ''.join(dnastr)
        modified.append(True)

      ref_sequences.append(dnastr)
      alt_sequences.append(alt)
      snp_inds.append(rel_pos)

  return ref_sequences, alt_sequences, snp_inds, modified

def get_row_score(row, tbifile, gerp=False):
  # gerp has two scores: neutral rate and RS score. phastcons/phylop only have one
  tbi_vals = next(tbifile.fetch(row['#Chrom'], row['Pos']-1, row['Pos']))
  if gerp:
    tbi_chr, tbi_pos, tbi_nscore, tbi_rsscore = tbi_vals.split('\t')
    tbi_nscore = float(tbi_nscore)
    tbi_rsscore = float(tbi_rsscore)
    tbi_pos = int(tbi_pos)
    assert tbi_pos == row['Pos']
    return tbi_nscore, tbi_rsscore
  else:
    tbi_chr, tbi_pos, tbi_score = tbi_vals.split('\t')
    tbi_score = float(tbi_score)
    tbi_pos = int(tbi_pos)
    assert tbi_pos == row['Pos']
    return tbi_score

def get_cons_scores(df):
  phastcon_file = 'data/remote_data/phastCons/primates_nohuman.tsv.gz'
  phylop_file = 'data/remote_data/phyloP/primates_nohuman.tsv.gz'
  gerp_file = 'data/remote_data/Gerp/gerp_scores.tsv.gz'
  ph_scores = []
  php_scores = []
  gerpn_scores = []
  gerprs_scores = []
  with pysam.Tabixfile(phastcon_file) as phfile:
    for i, (ix, row) in enumerate(df.iterrows()):
      score = get_row_score(row, phfile)
      ph_scores.append(score)

  with pysam.Tabixfile(phylop_file) as phpfile:
    for i, (ix, row) in enumerate(df.iterrows()):
      score = get_row_score(row, phpfile)
      php_scores.append(score)

  with pysam.Tabixfile(gerp_file) as gerpfile:
    for i, (ix, row) in enumerate(df.iterrows()):
      nscore, rsscore = get_row_score(row, gerpfile, gerp=True)
      gerpn_scores.append(nscore)
      gerprs_scores.append(rsscore)

  return ph_scores, php_scores, gerpn_scores, gerprs_scores


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

def snpfeats_from_df(df, seqlen=None, seqfeatextractor='deepsea',
                     compfeattype='absdiff', all_layers=False,
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
