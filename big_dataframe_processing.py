run_local = True


#

import os
import pickle
import sys
import time

if run_local:
    base_input_path = '.'
    base_output_path = '.'
    train_name_root = 'train'
    test_name_root = 'test'
    submission_name_root = 'sample_submission'

else:
    base_input_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
    base_output_path = '.'
    train_name_root = 'train'
    test_name_root = 'test'
    submission_name_root = 'sample_submission'

def make_csv_index_file(csv_path, cache_path):
    print(f'Reading file to get line offsets: {csv_path}')
    start_time = time.time()
    eol_byte_offsets = []
    with open(csv_path, 'rb') as fd: # in text mode file.tell() gives strange numbers apparently
        for line in fd:
            offset_now = fd.tell()
            eol_byte_offsets.append(offset_now)
    if len(eol_byte_offsets) > 0:
        eol_byte_offsets.pop() # Don't want off-end-of-file offset for 'next' row
    print(f'Scan took {time.time() - start_time} s')
    print(f'List len={len(eol_byte_offsets)}, memory size={sys.getsizeof(eol_byte_offsets)}')
    # Size of list in memory only a touch bigger than 8 bytes/element so looks OK to use directly

    start_time = time.time()
    with open(cache_path, 'wb') as fd:
        pickle.dump(eol_byte_offsets, fd)
    print(f'Cached offsets written to {cache_path} in {time.time() - start_time} s')

    return eol_byte_offsets

for name_root in [test_name_root, submission_name_root, train_name_root]:
    in_path = os.path.join(base_input_path,  name_root + '.csv')
    cache_path = os.path.join(base_output_path, name_root + '.pkl')

    offsets = make_csv_index_file(in_path, cache_path)

    read_from_proportion = 0.9
    useful_rows = len(offsets) - 1
    read_from_row_idx = int(read_from_proportion * useful_rows)
    print(f'Reading from row {read_from_row_idx} in {in_path}')
    start_time = time.time()
    with open(in_path, 'rb') as fd:
        start_byte_offset = offsets[read_from_row_idx]
        end_byte_offset = offsets[read_from_row_idx + 1]
        fd.seek(start_byte_offset)
        raw_data = fd.read(end_byte_offset - start_byte_offset)
        row_str = raw_data.decode("utf-8")
        print(f'Row read took {time.time() - start_time} s, start of row: {row_str[:50]}')
