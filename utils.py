import re
import csv
import pysam
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from constants import LOCS, LOCS_V2
from collections import defaultdict
from validate import score_preds, pr


def make_plots(cvdf_chunk, col='PredValue'):
  fig, ((ax_roc, ax_prc), (ax_value, ax_conf)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
  cv_grpd = cvdf_chunk.groupby('regulatory_element')
  colors = cm.Set1(np.linspace(0, 1, len(cv_grpd)))
  fpr, tpr, thresholds, auroc = make_roc_curve(ax_roc, cvdf_chunk, fill=True, label='overall', color='k')
  precision, recall, thresholds, auprc = pr_curve(ax_prc, cvdf_chunk, fill=True)
  for i, (name, grp) in enumerate(cv_grpd):
    pr_curve(ax_prc, grp, color=colors[i], fill=False)
    make_roc_curve(ax_roc, grp, label=name, color=colors[i], alpha=.5)
  auroc_axes(ax_roc, auroc)
  auprc_axes(ax_prc, auprc)
  return (
      (fpr, tpr, thresholds, auroc),
      (precision, recall, thresholds, auprc),
      make_value_plot(ax_value, cv_grpd, colors),
      make_conf_plot(ax_conf, cv_grpd, colors))


def make_roc_curve(ax, cvdf_chunk, color, col='PredValue', fill=False, **kwargs):
  """Makes a ROC plot of significant effect prediction."""
  fpr, tpr, thresholds = roc_curve(cvdf_chunk['class'].abs(), cvdf_chunk[col].abs(),
                                   pos_label=1)
  auroc = auc(fpr, tpr)
  ax.plot(fpr, tpr, color=color, **kwargs)
  if fill:
    ax.fill_between(fpr, tpr, step='post', alpha=0.2, color=color)
  return fpr, tpr, thresholds, auroc


def pr_curve(ax, cvdf_chunk, color='k', col='PredValue', fill=False):
  precision, recall, thresholds, auprc = pr(cvdf_chunk, col)
  ax.step(recall, precision, color=color, alpha=0.2, where='post')
  if fill:
    ax.fill_between(recall, precision, step='post', alpha=0.2, color=color)
  return precision, recall, thresholds, auprc


def auprc_axes(ax, auprc):
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_title('AUPRC: {:.2f}'.format(auprc))


def auroc_axes(ax, auroc):
  ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', alpha=.7)
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('AUROC: {:.2f}'.format(auroc))
  ax.legend(loc="lower right")


def make_value_plot(ax, cv_grpd, colors):
  """Makes a scatter plot of predicted vs. observed values."""
  for i, (name, grp) in enumerate(cv_grpd):
      ax.scatter(grp['Value'], grp['PredValue'], label=name, color=colors[i], alpha=.2)
  # ax.legend()
  ax.set_xlabel('value')
  ax.set_ylabel('predicted')
  return ax


def make_conf_plot(ax, cv_grpd, colors):
  """Makes a scatter plot of predicted vs. observed values."""
  for i, (name, grp) in enumerate(cv_grpd):
      ax.scatter(grp['Confidence'], grp['PredConfidence'], label=name, color=colors[i], alpha=.2)
  # ax.legend()
  ax.set_xlabel('confidence')
  ax.set_ylabel('predicted')
  return ax


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

def get_seqs_and_inds(df, use_modified=True, v=2):
  refs = []
  alts = []
  inds = []
  if v==2:
    base_seq_dict = load_base_seqs('data/cagi5_mpra/base_seqs_v2.csv')
  else:
    base_seq_dict = load_base_seqs()
  if 'regulatory_element' not in df.columns:
    df['regulatory_element'] = df['Promoter_Enhancer'] 
  for (ix, row) in df.iterrows():
    if re.match('release', row['regulatory_element']):
      reg_el_code = row['regulatory_element'][8:]
    else:
      reg_el_code = row['regulatory_element']
    if re.match('TERT', reg_el_code):
      reg_el_code = 'TERT'
    if re.match('MYC', reg_el_code):
      reg_el_code = 'MYCrs6983267MOD'
    if use_modified and reg_el_code+'MOD' in base_seq_dict:
      reg_el_code = reg_el_code + 'MOD'
    seq = base_seq_dict[reg_el_code]['seq']
    start = base_seq_dict[reg_el_code]['start']

    rel_pos = row['Pos'] - start - 1
    # print(row['Pos'], start)
    # if rel_pos == -1:
    #   # first position in SORT1 enhancer wasn't included previously
    #   seq = row['Ref'] + seq
    #   rel_pos += 1 # i.e. rel_pos is 0 now

    

    # if rel_pos == len(seq):
    #   seq += row['Ref']
      # print(rel_pos, len(seq))

    if use_modified:
      print("""Enhancer {}, relative SNP position {}, enhancer type {}
               retrieved ref nucleotide {}, dataset ref nucleotide {},
               alt nucleotide {}""".format(
                row['regulatory_element'], rel_pos, reg_el_code,
                seq[rel_pos], row['Ref'], row['Alt']))
      assert seq[rel_pos] == row['Ref']

    else:
      if rel_pos >= len(seq):
        print('Non matching end at index', ix)
        print('Enhancer {}, position {}, relative pos {}, seq len {}'.format(
          row['regulatory_element'], row['Pos'], rel_pos, len(seq)))
        continue
      elif rel_pos < 0:
        print('Non matching seq at index', ix)
        print('Enhancer {}, position {}, relative pos {}'.format(
          row['regulatory_element'], row['Pos'], rel_pos))
      elif seq[rel_pos] != row['Ref']:
        print('Non matching seq at index', ix)
        print('Enhancer {}, position {}, relative pos {}'.format(
          row['regulatory_element'], row['Pos'], rel_pos))

    alt = list(seq)
    alt[rel_pos] = row['Alt']
    alt = ''.join(alt)

    refs.append(seq)
    alts.append(alt)
    inds.append(rel_pos)

  return refs, alts, inds

def save_base_seqs_v2(df_train, df_test):
  fasta_file = 'data/remote_data/hg19.genome.fa'
  ref_sequences = set()
  ref_dict = {}
  df_test['#Chrom'] = df_test['Chrom']
  df_test['regulatory_element'] = 'release_' + df_test['Promoter_Enhancer']

  ref_train, alt_train, inds_train, modified_train = get_check_sequences_cagi5(df_train)
  ref_test, alt_test, inds_test, modified_test = get_check_sequences_cagi5(df_test)

  df = pd.concat([df_train, df_test])
  for (ix,row), ref, ind, mod in zip(df.iterrows(), ref_train+ref_test,
                                     inds_train+inds_test, modified_train+modified_test):

    if ref not in ref_sequences: # only add each base seq once
      suffix = ''
      reg_el_code = row['regulatory_element'][8:]
      if re.match('TERT', reg_el_code):
        reg_el_code = 'TERT'
      if re.match('MYC', reg_el_code):
        reg_el_code = 'MYCrs6983267'
      if mod:
        suffix += 'MOD'
        while True:
          if reg_el_code + suffix in ref_dict:
            suffix += 'MOD'
          else:
            break
      else:
        while True:
          if reg_el_code + suffix in ref_dict:
            suffix += 'V'
          else:
            break
      ref_dict[reg_el_code + suffix] = ref
      ref_sequences.add(ref)

  with open('data/cagi5_mpra/base_seqs_v2.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['regulatory_element', 'ref_sequence', 'start_pos'])
    for k, v in ref_dict.items():
      if re.search('MOD', k):
        reg_el_code = k.split('MOD')[0]
      elif re.search('V', k):
        reg_el_code = k.split('V')[0]
      else:
        reg_el_code = k
      if reg_el_code not in LOCS_V2:
        print(k, reg_el_code)
      seqstart = LOCS_V2[reg_el_code]['start']
      writer.writerow([k, v, seqstart])

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
        # this assumes that there's only one MOD per enhancer - not necessarily true
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

def get_check_sequences_cagi5(df, v=2):
  if v == 2:
    locs = LOCS_V2
  else:
    locs = LOCS
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
      if re.match('MYC', reg_el_code):
        reg_el_code = 'MYCrs6983267'
      seqstart = locs[reg_el_code]['start']
      seqend = locs[reg_el_code]['end']
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

def encode_sequences(sequences, seqlen=None, inds=None):
  # N.B. that the fact that Basenji, for example, does binned predictions should mean that
  # it actually can be applied to variable length sequences
  print('Encoding seqs of len {}'.format(500))
  preprocessed_seqs = []
  if seqlen is None:
    # just encode without padding
    preprocessed_seqs = sequences
  else:
    for i, seq in enumerate(sequences):
      if seqlen > len(seq):
        pad_left = (seqlen - len(seq))//2
        pad_right = seqlen - (len(seq) + pad_left)
        seq = 'N'*pad_left + seq + 'N'*pad_right
      else:
        snp_ind = inds[i]
        avail_right = len(seq) - snp_ind
        if avail_right > snp_ind:
          start = 0
          end = seqlen
        else:
          start = len(seq) - 500
          end = len(seq)
        seq = seq[start:end]
      assert len(seq) == seqlen
        # we want to make it roughly central b.c. the local info is the most important
        # raise Exception('trying to create seqs of length {} but received length {}'.format(seqlen, len(seq)))
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
