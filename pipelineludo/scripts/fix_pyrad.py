import pandas as pd

df_train = pd.read_excel('data/pyrad_correct/train_set.xlsx')
df_test = pd.read_excel('data/pyrad_correct/test_set.xlsx')
df_xval = pd.read_excel('data/pyrad_correct/uoc_set.xlsx')

# only keep the columns called Subject or that contain the word 'vanilla' in the column name
df_train = df_train[df_train.columns[(df_train.columns == 'Subject') | df_train.columns.str.contains('vanilla')]]
df_test = df_test[df_test.columns[(df_test.columns == 'Subject') | df_test.columns.str.contains('vanilla')]]
df_xval = df_xval[df_xval.columns[(df_xval.columns == 'Subject') | df_xval.columns.str.contains('vanilla')]]

# check that all three dataset have the same columns
assert (df_train.columns == df_test.columns).all() and (df_test.columns == df_xval.columns).all(), "The three datasets do not have the same columns."

# concatenate the three datasets
df_all = pd.concat([df_train, df_test, df_xval])

# convert all feature except Subject to float array and put it in a new column called mod2
df_all['mod2'] = df_all.iloc[:, 1:].apply(lambda x: x.astype(float).tolist(), axis=1)

# if any of the elements in mod2 are nan replace the list with None
df_all['mod2'] = df_all['mod2'].apply(lambda x: None if any(pd.isna(x)) else x)

# delete all columns except Subject and mod2
df_all = df_all[['Subject', 'mod2']]

# ----------------------------

# read data/features_dataset_radpy.parquet
pt = pd.read_parquet('data/features_dataset_radpy.parquet')

# drop mod2 column
pt = pt.drop(columns=['mod2'])

# merge on Subject-slide and replace mod2 with the mod2 column from df_all
pt = pt.merge(df_all, left_on='slide', right_on='Subject', how='left')

# drop Subject and reorder columns
pt = pt[['slide', 'mod1', 'mod2', 'mod3', 'mod4']]

# save the dataset
pt.to_parquet('data/features_dataset_radpy_fixed.parquet')






