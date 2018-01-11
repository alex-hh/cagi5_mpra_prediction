import pysam

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