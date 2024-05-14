import polars as pl
import time

for name_root in ['test', 'sample_submission', 'train']:
    in_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim/' + name_root + '.csv'
    print('Reading', in_path)
    
    # https://stackoverflow.com/questions/75523498/python-polars-how-to-get-the-row-count-of-a-dataframe
    start_time = time.time()
    lazy_df = pl.scan_csv(in_path)
    length = lazy_df.select(pl.len()).collect().item()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Length: ', length, 'Seconds: ', elapsed_time)