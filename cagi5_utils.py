import pandas as pd

def get_breakpoint_df(df):
  prev_pos = 9999999999999
  prev_el = None
  is_break = []
  
  regelcol = 'regulatory_element'
  if 'regulatory_element' not in df.columns:
    regelcol = 'Promoter_Enhancer'
  df = df.sort_values([regelcol, 'Pos']) # needed because the two TERT cell types are interspersed in the test dset
  for i, (ix, row) in enumerate(df.iterrows()):
    if row['Pos'] != prev_pos and row['Pos'] - prev_pos != 1:
      if is_break:
          is_break[-1] = 'end'
#             if row[regelcol] != prev_el and is_break:
#                 is_break[-1] = True
      is_break.append('start')
    else:
      if i == len(df) -1:
        is_break.append('end')
      else:
        is_break.append('no')
    prev_pos = row['Pos']
    prev_el = row[regelcol]
  df['is_break'] = is_break
  
#     breakpoint_df = pd.DataFrame(df[df['is_break'].isin(['start', 'end'])])
  breakpoint_df = pd.DataFrame(df)

  prev_start = 999999999999
  prev_el = None

  lengths = []
  for ix, row in breakpoint_df.iterrows():
    if row['is_break'] == 'start':
      prev_start = row['Pos']
    if row['is_break'] == 'end':
      lengths.append(row['Pos']-prev_start+1)
    else:
      lengths.append(None)
  breakpoint_df['chunk_length'] = lengths

  return breakpoint_df

def get_chunk_counts(df):
  assert 'is_break' in df.columns
  chunk_counts = df[df['is_break']=='end'].groupby(['regulatory_element'])['chunk_length'].agg(['sum', 'count'])
  # chunk_counts['length_per_chunk'] = chunk_counts['sum'] / chunk_counts['count']
  # chunk_counts['n_chunk_val'] = np.ceil(chunk_counts['count'] / val_folds) # number of validation chunks per fold
  return chunk_counts