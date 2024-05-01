# LEAP competition with feature engineering

import os
import polars as pl


run_local = True
debug = True

if debug:
    max_train_rows = 2000
else:
    max_train_rows = 0 # all


if run_local:
    base_path = '.'
    train_path = os.path.join(base_path, 'train-top.csv')
    # Will have to inline imports on kaggle
    import column_info
else:
    base_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
    train_path = os.path.join(base_path, 'train.csv')

# Read in training data
pl_read_opts = {}
if max_train_rows > 0:
    # Limit number of rows we load, tolerates larger number than exist
    # TODO polars can do read_csv_batched
    pl_read_opts['n_rows'] = max_train_rows

train_df = pl.read_csv(train_path, **pl_read_opts)


# Add columns for new features
unexpanded_col_list = column_info.col_info_list

unexpanded_col_list.append(column_info.ColumnInfo(True, 'air_density', 'air density', 60, 'kg/m3'))
unexpanded_col_list.append(column_info.ColumnInfo(True, 'momentum_u', 'zonal momentum per unit volume',      60, '(kg.m/s)/m3')),
unexpanded_col_list.append(column_info.ColumnInfo(True, 'momentum_v', 'meridional momentum per unit volume', 60, '(kg.m/s)/m3')),

pass