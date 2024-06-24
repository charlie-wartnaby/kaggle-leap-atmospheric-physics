# Dump from my Kaggle notebook

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

num_rows = 10
for name_root in ['submission', 'test']:#, 'sample_submission']:
    #in_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim/' + name_root + '.csv'
    in_path = os.path.join('.', name_root + '.csv')
    df_top = pd.read_csv(in_path, nrows=num_rows)
    out_path = name_root + f'-top-{num_rows}.csv'
    df_top.to_csv(out_path)
