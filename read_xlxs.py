c5_df_dict = pd.read_excel('data/release.xlsx', sheetname=None, skiprows=6)
for k, df in c5_df_dict.items():
  df['class'] = df.apply(lambda row: get_class(row), axis=1)
  df['regulatory_element'] = k
  
all_data = pd.concat(c5_df_dict.values(), ignore_index=True)
all_data.to_csv('data/cagi5_df.csv', index=False)