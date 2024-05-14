import polars as pl
import time

for name_root in ['test']: #, 'sample_submission', 'train']:
    in_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim/' + name_root + '.csv'
    print('Reading', in_path)
    
    # https://stackoverflow.com/questions/75523498/python-polars-how-to-get-the-row-count-of-a-dataframe
    print("Lazy scan of file")
    start_time = time.time()
    lazy_df = pl.scan_csv(in_path)
    print(f"... took {time.time() - start_time}")
    # length = lazy_df.select(pl.len()).collect().item()
    
    # Checking how long it takes to 'materialise' bits of dataframe 
    total_len = 625000 # from earlier experiment
    
    # Is it quicker to get earlier data once it has been forced to analyse near the end,
    # is a memory map preserved of where to find the rows?
    for start_offset_propn in [0.1, 0.9, 0.2, 0.5, 0.95]:
        start_offset = int(start_offset_propn * total_len)
        print("Materalising from offset:", start_offset, "proportion:", start_offset_propn)
        start_time = time.time()
        materialised_slice = lazy_df.slice(start_offset, 1000).collect()
        print(f"... took {time.time() - start_time}")
        print(materialised_slice.describe())
