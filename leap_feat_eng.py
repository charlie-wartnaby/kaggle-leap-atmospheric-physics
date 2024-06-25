#!/bin/env python

# (c) Charlie Wartnaby 2024
#
# This is my entry for this competition:
# https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim
#
# This uses a 1-D convolutional neural network approach in PyTorch,
# and/or a decisition tree approach using catboost.
#
# The original set of features is augmented using new ones inspired by
# the physics of air parcels which I hoped would get better results.
#
# See README.md for more
#
# These example notebooks for PyTorch and catboost got me started, thank you:
# https://www.kaggle.com/code/airazusta014/pytorch-nn/notebook
# https://www.kaggle.com/code/lonnieqin/leap-catboost-baseline
# https://www.kaggle.com/code/gogo827jz/multiregression-catboost-1-model-for-206-targets


# This can be run locally or the whole thing pasted into a single Kaggle notebook cell


import catboost
import copy
import gc
import io
import numpy as np
import os
import pickle
import polars as pl
import random
import re
import shutil
import sklearn.metrics
import sklearn.model_selection
import socket
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import Dataset, DataLoader
import warnings


# Settings
debug = False
do_test = True
is_rerun = False
do_analysis = True
do_train = True
do_feature_knockout = False
clear_batch_cache_at_start = False
scale_using_range_limits = False
use_float64 = False
model_type = "catboost"
emit_scaling_stats = False


if debug:
    max_train_rows = 1000
    max_test_rows  = 100
    caboost_batch_size = 100
    cnn_batch_size = 100
    catboost_batch_size = 100
    patience = 4
    train_proportion = 0.8
    max_epochs = 1
else:
    # Use very large numbers for 'all'
    max_train_rows = 800000
    max_test_rows  = 1000000000
    catboost_batch_size = 20000
    cnn_batch_size = 5000
    patience = 3 # was 5 but saving GPU quota
    train_proportion = 0.9
    max_epochs = 50

subset_base_row = 0

# For model parameters to form permutations of in hyperparameter search
# Each entry is 'param_name' : [list of values for that parameter]
multitrain_params = {}

show_timings = False # debug
batch_report_interval = 10
dropout_p = 0.1
initial_learning_rate = 0.001 # default 0.001
try_reload_model = is_rerun
clear_batch_cache_at_end = False # can save Kaggle quota by deleting there?
max_analysis_output_rows = 10000
min_std = 1e-30
np.random.seed(42)
random.seed(42)

machine = socket.gethostname()
is_kaggle = re.match(r"[a-f0-9]{12}", machine) # Kaggle e.g. machine "1e5e4ffe5117"
run_local = not is_kaggle

if machine == 'narg':
    # Small computer which can't cope with the full huge files
    train_root = 'train-top-5000'
    test_root = 'test-top-1000'
    submission_root = 'sample_submission-top-1000'
else:
    train_root = 'train'
    test_root = 'test'
    submission_root = 'sample_submission'

if run_local:
    base_path = '.'
    offsets_path = '.'
else:
    base_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
    offsets_path = '/kaggle/input/leap-atmospheric-physics-file-row-offsets'

save_backup_cache = (machine == 'greta')
train_path = os.path.join(base_path, train_root + '.csv')
train_offsets_path = os.path.join(offsets_path, train_root + '.pkl')
test_path = os.path.join(base_path, test_root + '.csv')
test_offsets_path = os.path.join(offsets_path, test_root + '.pkl')
submission_template_path = os.path.join(base_path, submission_root + '.csv')
analysis_df_path = 'r2_analysis.csv'
r2_ranking_path = 'r2_ranking.csv'
if debug:
    model_root_path = 'model_debug'
    epoch_counter_path = 'epochs_debug.txt'
    loss_log_path = 'loss_log_debug.csv'
    batch_cache_dir = 'batch_cache_debug'
else:
    model_root_path = 'model'
    epoch_counter_path = 'epochs.txt'
    loss_log_path = 'loss_log.csv'
    batch_cache_dir = 'batch_cache'
scaling_cache_filename = 'scaling_normalisation.pkl'
scaling_cache_path = os.path.join(batch_cache_dir, scaling_cache_filename)

feature_knockout_path = 'feature_knockout.csv'
stopfile_path = 'stop.txt'
cnn_analysis_data_path = 'cnn_analysis.data.pkl' # Including R2 score per output column
catboost_analysis_data_path = 'catboost_analysis_data.pkl'

# Use smallest common size for caching, so that we don't do unnecessarily
# large numpy operations when loading cache batches
cache_batch_size = min(cnn_batch_size, catboost_batch_size)
test_batch_size = cache_batch_size

def main():
    entry_clean_up()

    # Read in training data
    print('Loading training HoloFrame...')
    train_hf = HoloFrame(train_path, train_offsets_path)

    num_train_rows = min(len(train_hf), max_train_rows)
    max_batch_size = max(cnn_batch_size, catboost_batch_size)
    num_biggest_batches = num_train_rows // max_batch_size
    num_train_rows = num_biggest_batches * max_batch_size # Now exact multiple of batch sizes
    assert(subset_base_row + num_train_rows <= len(train_hf))

    col_data = form_col_data()
    expand_vector_col_info(col_data)

    if do_feature_knockout:
        param_permutations = range(len(col_data.unexpanded_input_col_names))
        with open(feature_knockout_path, 'w') as fd:
            fd.write(f"i,best_val_loss,Variable,Description\n")
    else:
        param_permutations = expand_multitrain_permutations()

    submission_weights_current, submission_weights_old = load_submission_weights(col_data)

    # Cache normalisation data needed for any rerun later
    if os.path.exists(scaling_cache_path):
        print("Opening previous scalings...")
        with open(scaling_cache_path, 'rb') as fd:
            scaling_data = pickle.load(fd)
        # Everything should already be cached
    else:
        scaling_data = preprocess_training_data(train_hf, num_train_rows, col_data, submission_weights_current, submission_weights_old)
    if emit_scaling_stats:
        write_scaling_stats_to_file(scaling_data, col_data)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if do_train:
        exec_data, bad_r2_output_names = training_loop(train_hf, num_train_rows, col_data, scaling_data, submission_weights_old,
                                                        param_permutations, device)

    if do_test:
        test_submission(col_data, scaling_data, exec_data, bad_r2_output_names, device, submission_weights_current, submission_weights_old)

    exit_clean_up()


def expand_multitrain_permutations():
    multitrain_keys = list(multitrain_params.keys())
    if len(multitrain_keys) < 1:
        param_permutations = [{}]
    else:
        permutation_indices = [0] * len(multitrain_keys)
        current_key_idx = 0
        param_permutations = []
        while (True):
            permutation = {}
            for param_idx in range(len(multitrain_keys)):
                param = multitrain_keys[param_idx]
                permutation[param] = multitrain_params[param][permutation_indices[param_idx]]
            param_permutations.append(permutation)
            if permutation_indices[current_key_idx] < len(multitrain_params[multitrain_keys[current_key_idx]]) - 1:
                permutation_indices[current_key_idx] += 1
                continue
            while (current_key_idx < len(multitrain_keys) and
                permutation_indices[current_key_idx] >= (len(multitrain_params[multitrain_keys[current_key_idx]]) - 1)):
                current_key_idx += 1
                for i in range(current_key_idx):
                    permutation_indices[i] = 0
            if current_key_idx >= len(multitrain_keys):
                break
            else:
                permutation_indices[current_key_idx] += 1
                current_key_idx = 0

    if is_rerun and len(param_permutations) > 1:
        print("Cannot do multitraining experiment and resume previous run")
        sys.exit(1)

    return param_permutations


def entry_clean_up():
    if os.path.exists(stopfile_path):
        print("Stop file detected on entry, deleting it")
        os.remove(stopfile_path)
    if not is_rerun and os.path.exists(epoch_counter_path):
        os.remove(epoch_counter_path)
    if not is_rerun and os.path.exists(loss_log_path):
        os.remove(loss_log_path)
    if clear_batch_cache_at_start and os.path.exists(batch_cache_dir):
        if save_backup_cache:
            # From times I've regretted deleting cache data...
            backup_cache_dir = batch_cache_dir + '_bak'
            if os.path.exists(backup_cache_dir):
                shutil.rmtree(backup_cache_dir)
            print(f'Saving old cache files to {backup_cache_dir} just in case...')
            os.rename(batch_cache_dir, backup_cache_dir)
        else:
            print('Deleting any previous batch cache files...')
            shutil.rmtree(batch_cache_dir)
    os.makedirs(batch_cache_dir, exist_ok=True)


class HoloFrame:
    """Manage data extraction from large .csv file with random access
    using precomputed byte offsets of each text row"""
    def __init__(self, csv_path, offsets_path):
        with open(offsets_path, 'rb') as offsets_fd:
            self.offset = pickle.load(offsets_fd)
        self.csv_fd = open(csv_path, 'rb')
        self.raw_headings = self.csv_fd.read(self.offset[0])
        pass        

    def __len__(self):
        """Length of CSV file in terms of data rows"""
        return len(self.offset)
    
    def get_slice(self, start_row_idx, end_row_idx):
        """Create Polars dataframe from headings + slice subset of rows"""
        start_byte_offset = self.offset[start_row_idx]
        # Treat -ve index like -ve slice index relative to end,
        # and 0 index like omitted index (start or end)
        if end_row_idx < 0:
            end_row_idx = len(self) + end_row_idx
        elif end_row_idx == 0:
            end_row_idx = len(self)
        if start_row_idx < 0:
            start_row_idx = len(self) + start_row_idx
        self.csv_fd.seek(start_byte_offset)
        if end_row_idx >= len(self):
            # There is no offset for the next row
            raw_data = self.csv_fd.read()
        else:    
            end_byte_offset = self.offset[end_row_idx]
            raw_data = self.csv_fd.read(end_byte_offset - start_byte_offset)
        complete_slice_csv = self.raw_headings + raw_data
        slice_df = pl.read_csv(complete_slice_csv)
        return slice_df



# Altitude levels in hPa from ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc
level_pressure_hpa = [0.07834781133863082, 0.1411083184744011, 0.2529232969453412, 0.4492506351686618, 0.7863461614709879, 1.3473557602677517, 2.244777286900205, 3.6164314830257718, 5.615836425337344, 8.403253219853443, 12.144489352066294, 17.016828024303006, 23.21079811610005, 30.914346261995327, 40.277580662953575, 51.37463234765765, 64.18922841394662, 78.63965761131159, 94.63009200213703, 112.09127353988006, 130.97780378937776, 151.22131809551237, 172.67390465199267, 195.08770981962772, 218.15593476138105, 241.60037901222947, 265.2585152868483, 289.12232222921756, 313.31208711045167, 338.0069992368819, 363.37349177951705, 389.5233382784413, 416.5079218282233, 444.3314120123719, 472.9572063769364, 502.2919169181905, 532.1522731583445, 562.2393924639011, 592.1492760575118, 621.4328411158061, 649.689897132655, 676.6564846051039, 702.2421877859194, 726.4985894989197, 749.5376452869328, 771.4452171682528, 792.2342599534793, 811.8566751313328, 830.2596431972574, 847.4506530638328, 863.5359020075301, 878.7158746040692, 893.2460179738746, 907.3852125876941, 921.3543974831824, 935.3167171670306, 949.3780562075774, 963.5995994020714, 978.013432382012, 992.6355435925217]
# Convert to Pa and reshape to be convenient later
level_pressure_pa_np = 100.0 * np.array(level_pressure_hpa, dtype=np.float32).reshape(1, -1)

# Manage columns as described in competition
num_atm_levels = 60
class ColumnInfo():
    def __init__(self, is_input, name, description, dimension, units='', first_useful_idx=0):
        self.is_input         = is_input
        self.name             = name
        self.description      = description
        self.dimension        = dimension
        self.units            = units
        self.first_useful_idx = first_useful_idx

class ColumnMetadata():
    def __init__(self):
        # Will assign members directly as loose struct
        pass

class ScalingMetadata():
    def __init__(self):
        # Will assign members directly as loose struct
        pass


def form_col_data():
    # Data column metadata from https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data
    unexpanded_col_list = [
        #         input?   Name                Description                                    Dimension   Units   Useful-from-idx
        ColumnInfo(True,  'state_t',          'air temperature',                                     60, 'K'      ),
        ColumnInfo(True,  'state_q0001',      'specific humidity',                                   60, 'kg/kg'  ),
        ColumnInfo(True,  'state_q0002',      'cloud liquid mixing ratio',                           60, 'kg/kg'  ),
        ColumnInfo(True,  'state_q0003',      'cloud ice mixing ratio',                              60, 'kg/kg'  ),
        ColumnInfo(True,  'state_u',          'zonal wind speed',                                    60, 'm/s'    ),
        ColumnInfo(True,  'state_v',          'meridional wind speed',                               60, 'm/s'    ),
        ColumnInfo(True,  'state_ps',         'surface pressure',                                     1, 'Pa'     ),
        ColumnInfo(True,  'pbuf_SOLIN',       'solar insolation',                                     1, 'W/m2'   ),
        ColumnInfo(True,  'pbuf_LHFLX',       'surface latent heat flux',                             1, 'W/m2'   ),
        ColumnInfo(True,  'pbuf_SHFLX',       'surface sensible heat flux',                           1, 'W/m2'   ),
        ColumnInfo(True,  'pbuf_TAUX',        'zonal surface stress',                                 1, 'N/m2'   ),
        ColumnInfo(True,  'pbuf_TAUY',        'meridional surface stress',                            1, 'N/m2'   ),
        ColumnInfo(True,  'pbuf_COSZRS',      'cosine of solar zenith angle',                         1           ),
        ColumnInfo(True,  'cam_in_ALDIF',     'albedo for diffuse longwave radiation',                1           ),
        ColumnInfo(True,  'cam_in_ALDIR',     'albedo for direct longwave radiation',                 1           ),
        ColumnInfo(True,  'cam_in_ASDIF',     'albedo for diffuse shortwave radiation',               1           ),
        ColumnInfo(True,  'cam_in_ASDIR',     'albedo for direct shortwave radiation',                1           ),
        ColumnInfo(True,  'cam_in_LWUP',      'upward longwave flux',                                 1, 'W/m2'   ),
        ColumnInfo(True,  'cam_in_ICEFRAC',   'sea-ice areal fraction',                               1           ),
        ColumnInfo(True,  'cam_in_LANDFRAC',  'land areal fraction',                                  1           ),
        ColumnInfo(True,  'cam_in_OCNFRAC',   'ocean areal fraction',                                 1           ),
        ColumnInfo(True,  'cam_in_SNOWHLAND', 'snow depth over land',                                 1, 'm'      ),
        ColumnInfo(True,  'pbuf_ozone',       'ozone volume mixing ratio',                           60, 'mol/mol'),
        ColumnInfo(True,  'pbuf_CH4',         'methane volume mixing ratio',                         60, 'mol/mol'),
        ColumnInfo(True,  'pbuf_N2O',         'nitrous oxide volume mixing ratio',                   60, 'mol/mol'),
        ColumnInfo(False, 'ptend_t',          'heating tendency',                                    60, 'K/s'    ),
        ColumnInfo(False, 'ptend_q0001',      'moistening tendency',                                 60, 'kg/kg/s', 12),
        ColumnInfo(False, 'ptend_q0002',      'cloud liquid mixing ratio change over time',          60, 'kg/kg/s', 12),
        ColumnInfo(False, 'ptend_q0003',      'cloud ice mixing ratio change over time',             60, 'kg/kg/s', 12),
        ColumnInfo(False, 'ptend_u',          'zonal wind acceleration',                             60, 'm/s2'   , 12),
        ColumnInfo(False, 'ptend_v',          'meridional wind acceleration',                        60, 'm/s2'   , 12),
        ColumnInfo(False, 'cam_out_NETSW',    'net shortwave flux at surface',                        1, 'W/m2'   ),
        ColumnInfo(False, 'cam_out_FLWDS',    'ownward longwave flux at surface',                     1, 'W/m2'   ),
        ColumnInfo(False, 'cam_out_PRECSC',   'snow rate (liquid water equivalent)',                  1, 'm/s'    ),
        ColumnInfo(False, 'cam_out_PRECC',    'rain rate',                                            1, 'm/s'    ),
        ColumnInfo(False, 'cam_out_SOLS',     'downward visible direct solar flux to surface',        1, 'W/m2'   ),
        ColumnInfo(False, 'cam_out_SOLL',     'downward near-infrared direct solar flux to surface',  1, 'W/m2'   ),
        ColumnInfo(False, 'cam_out_SOLSD',    'downward diffuse solar flux to surface',               1, 'W/m2'   ),
        ColumnInfo(False, 'cam_out_SOLLD',    'downward diffuse near-infrared solar flux to surface', 1, 'W/m2'   ),
    ]

    # Add columns for new features
    unexpanded_col_list.append(ColumnInfo(True, 'pressure',             'air pressure',                          60, 'N/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'density',              'air density',                           60, 'kg/m3'      ))
    unexpanded_col_list.append(ColumnInfo(True, 'recip_density',        'reciprocal air density',                60, 'm3/kg'      ))
    unexpanded_col_list.append(ColumnInfo(True, 'recip_ice_cloud',      'reciprocal ice cloud',                  60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'recip_water_cloud',    'reciprocal water cloud',                60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'wind_rh_prod',         'wind-rel humidity product',             60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'wind_cloud_prod',      'wind-total cloud product',              60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'total_cloud',          'total ice + liquid cloud',              60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'total_cloud_density',  'total cloud density product',           60, 'kg/m3'      ))
    unexpanded_col_list.append(ColumnInfo(True, 'total_gwp',            'total global warming potential',        60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'sensible_flux_gwp_prod','total GWP - sensible heat flux prod',  60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'up_lw_flux_gwp_prod',  'total GWP - upward longwave flux prod', 60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'abs_wind',             'abs wind magnitude',                    60, 'm/s'        ))
    unexpanded_col_list.append(ColumnInfo(True, 'abs_momentum',         'abs momentum per unit volume',          60, '(kg.m/s)/m3'))
    unexpanded_col_list.append(ColumnInfo(True, 'abs_stress',           'abs stress magnitude',                  60, 'N/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'rel_humidity',         'relative humidity (proportion)',        60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'recip_rel_humidity',   'reciprocal relative humidity',          60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'buoyancy',             'Beucler buoyancy metric',               60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'up_integ_tot_cloud',   'ground-up integral of total cloud',     60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'down_integ_tot_cloud', 'sky-down integral of total cloud',      60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'lat_heat_div_density', 'latent heat flux divided by density',   60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'sen_heat_div_density', 'sensible heat flux divided by density', 60               ))
    unexpanded_col_list.append(ColumnInfo(True, 'vert_insolation',      'zenith-adjusted insolation',             1, 'W/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'direct_sw_absorb',     'direct shortwave absorbance',            1, 'W/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'diffuse_sw_absorb',    'diffuse shortwave absorbance',           1, 'W/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'direct_lw_absorb',     'direct longwave absorbance',             1, 'W/m2'       ))
    unexpanded_col_list.append(ColumnInfo(True, 'diffuse_lw_absorb',    'diffuse longwave absorbance',            1, 'W/m2'       ))


    col_data = ColumnMetadata()
    unexpanded_col_names = [col.name for col in unexpanded_col_list]
    col_data.unexpanded_cols_by_name = dict(zip(unexpanded_col_names, unexpanded_col_list))
    col_data.unexpanded_input_col_names = [col.name for col in unexpanded_col_list if col.is_input]
    col_data.unexpanded_output_col_names = [col.name for col in unexpanded_col_list if not col.is_input]
    col_data.unexpanded_output_vector_col_names = [col.name for col in unexpanded_col_list if not col.is_input and col.dimension > 1]
    col_data.unexpanded_output_scalar_col_names = [col.name for col in unexpanded_col_list if not col.is_input and col.dimension <= 1]

    col_data.input_trick_names = ["state_q0002", "state_q0003"]

    # Form set of names to not compute outputs for according to competition
    # description; may add our own poor ones to this set later
    col_data.zero_or_bad_cols_by_name = {}

    col_data.output_expanded_first_idx_by_name = {}
    current_idx = 0
    for col_name in col_data.unexpanded_output_col_names:
        col = col_data.unexpanded_cols_by_name[col_name]
        col_data.output_expanded_first_idx_by_name[col_name] = current_idx
        for i in range(col.first_useful_idx):
            expanded_name = f'{col.name}_{i}'
            col_data.zero_or_bad_cols_by_name[expanded_name] = True
        current_idx += col.dimension

    col_data.feature_idx_by_name = {}
    for i, name in enumerate(col_data.unexpanded_input_col_names):
        col_data.feature_idx_by_name[name] = i

    if do_feature_knockout:
        col_data.cnn_input_feature_idx = range(len(col_data.unexpanded_input_col_names))
        col_data.catboost_input_feature_idx = col_data.cnn_input_feature_idx
        col_data.cnn_subset_idx_by_name = col_data.feature_idx_by_name
        col_data.catboost_subset_idx_by_name = col_data.feature_idx_by_name
    else:
        # Including most for CNN, eliminated some replaced by derived features:
        current_cnn_knockout_features = set(['pbuf_COSZRS',
                                                'cam_in_ALDIF',
                                                'cam_in_ALDIR',
                                                'cam_in_ASDIF',
                                                'cam_in_ASDIR',
                                                'pbuf_ozone',
                                                'pbuf_CH4',
                                                'pbuf_N2O'])
        # List came from feature knockout ranking actually from CNN experiments:
        current_catboost_use_features = set(['state_q0002', 'state_q0003', 'rel_humidity', 'state_v',
                                            'state_t', 'cam_in_LANDFRAC', 'diffuse_lw_absorb',
                                            'abs_momentum', 'direct_sw_absorb','buoyancy',
                                            'up_integ_tot_cloud', 'cam_in_ALDIR', 'recip_rel_humidity',
                                            'state_q0001', 'cam_in_OCNFRAC', 'pbuf_LHFLX', #'pbuf_CH4',
                                            'state_ps', 'cam_in_ICEFRAC', 'pressure', 'pbuf_COSZRS',
                                            'density',
                                            # Trying a few more, just short of memory exhaustion
                                            'state_u', 'pbuf_TAUX', 'pbuf_TAUY',
                                            'recip_density', 'recip_ice_cloud', 'recip_water_cloud',
                                            'total_cloud_density', 'sensible_flux_gwp_prod',
                                            'up_lw_flux_gwp_prod', 'abs_wind', 'abs_momentum', 'abs_stress',
                                            'down_integ_tot_cloud', 'lat_heat_div_density',
                                            'vert_insolation', 'diffuse_sw_absorb', 'direct_lw_absorb',
                                            ])
        
        col_data.cnn_input_feature_idx = []
        col_data.catboost_input_feature_idx = []
        col_data.cnn_subset_idx_by_name = {}
        col_data.catboost_subset_idx_by_name = {}
        for idx, name in enumerate(col_data.unexpanded_input_col_names):
            if name not in current_cnn_knockout_features:
                col_data.cnn_subset_idx_by_name[name] = len(col_data.cnn_input_feature_idx)
                col_data.cnn_input_feature_idx.append(idx)
            if name in current_catboost_use_features:
                col_data.catboost_subset_idx_by_name = len(col_data.catboost_input_feature_idx)
                col_data.catboost_input_feature_idx.append(idx)

    return col_data


def expand_and_add_cols(col_list, cols_by_name, col_names):
    for col_name in col_names:
        col_info = cols_by_name[col_name]
        if col_info.dimension <= 1:
            col_list.append(col_info)
        else:
            for i in range(col_info.dimension):
                col_info = copy.copy(cols_by_name[col_name])
                col_info.name = col_info.name + f'_{i}'
                col_list.append(col_info)

def expand_vector_col_info(col_data):
    col_data.expanded_col_list = []
    expand_and_add_cols(col_data.expanded_col_list, col_data.unexpanded_cols_by_name, col_data.unexpanded_input_col_names)
    expand_and_add_cols(col_data.expanded_col_list, col_data.unexpanded_cols_by_name, col_data.unexpanded_output_col_names)

    expanded_names = [col.name for col in col_data.expanded_col_list]
    col_data.expanded_cols_by_name = dict(zip(expanded_names, col_data.expanded_col_list))
    col_data.expanded_names_input = [col.name for col in col_data.expanded_col_list if col.is_input]
    col_data.expanded_names_output = [col.name for col in col_data.expanded_col_list if not col.is_input]

    col_data.num_all_outputs_as_vectors = len(col_data.unexpanded_output_col_names)
    col_data.num_pure_vector_outputs = len(col_data.unexpanded_output_vector_col_names)
    col_data.num_scalar_outputs = len(col_data.unexpanded_output_scalar_col_names)


# Functions to compute saturation pressure at given temperature taken from
# https://colab.research.google.com/github/tbeucler/CBRAIN-CAM/blob/master/Climate_Invariant_Guide.ipynb#scrollTo=1Hsy9p4Ghe-G
# required to compute relative humidity, which generalises much better than
# specific humidity according to Beucler et al 2021:

# Constants for the Community Atmosphere Model
DT = 1800.
L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_F = L_I
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
RHO_L = 1e3

def eliq(T):
    """
    Function taking temperature (in K) and outputting liquid saturation
    pressure (in hPa) using a polynomial fit
    """
    a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,
                              0.206739458e-7,0.302950461e-5,0.264847430e-3,
                              0.142986287e-1,0.443987641,6.11239921]);
    c_liq = -80
    T0 = 273.16
    return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))

def eice(T):
    """
    Function taking temperature (in K) and outputting ice saturation
    pressure (in hPa) using a polynomial fit
    """
    a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,
                      0.188439774e-1,0.503160820,6.11147274]);
    c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
    T0 = 273.16
    return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*\
                       (c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))


def RH_from_climate(specific_humidity, temperature, air_pressure_Pa):
    
    # # 0) Extract specific humidity, temperature, and air pressure
    # specific_humidity = data['vars'][:,:30] # in kg/kg
    # temperature = data['vars'][:,30:60] # in K

    # P0 = 1e5 # Mean surface air pressure (Pa)
    # near_surface_air_pressure = data['vars'][:,60]
    # # Formula to calculate air pressure (in Pa) using the hybrid vertical grid
    # # coefficients at the middle of each vertical level: hyam and hybm
    # air_pressure_Pa = np.outer(near_surface_air_pressure**0,P0*hyam) + \
    # np.outer(near_surface_air_pressure,hybm)

    # 1) Calculating saturation water vapor pressure
    T0 = 273.16 # Freezing temperature in standard conditions
    T00 = 253.16 # Temperature below which we use e_ice
    omega = (temperature - T00) / (T0 - T00)
    omega = np.maximum( 0, np.minimum( 1, omega ))

    esat =  omega * eliq(temperature) + (1-omega) * eice(temperature)

    # 2) Calculating relative humidity
    Rd = 287 # Specific gas constant for dry air
    Rv = 461 # Specific gas constant for water vapor

    # We use the `values` method to convert Xarray DataArray into Numpy ND-Arrays
    #return Rv/Rd * air_pressure_Pa/esat.values * specific_humidity.values
    return Rv/Rd * air_pressure_Pa/esat * specific_humidity

def esat(T):
    T0 = 273.16
    T00 = 253.16
    omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

    return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))

def qv(T,RH, p): #P0,PS,hyam,hybm):
    R = 287
    Rv = 461
    #S = PS.shape
    # p = 1e5 * np.tile(hyam,(S[0],1))+np.transpose(np.tile(PS,(30,1)))*np.tile(hybm,(S[0],1))

    return R*esat(T)*RH/(Rv*p)

def qsat(T, p): #,P0,PS,hyam,hybm):
    return qv(T,1,p) # P0,PS,hyam,hybm)

# Beucler et al plume buoyance metric from bmse_calc() wrapped in class T2BMSENumpy
# https://colab.research.google.com/github/tbeucler/CBRAIN-CAM/blob/master/Climate_Invariant_Guide.ipynb#scrollTo=0S6W988UaG6p&line=1&uniqifier=1
# "Source code for the moist thermodynamics library"
# qv = specific humidity from comment on class QV2RHNumpy
# P0, PS clues from "p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)"
# "P0 = 1e5 # Mean surface air pressure (Pa)"
# "surface pressure `PS` in hPa"
# "hyam=hyam, hybm=hybm, # Arrays to define mid-levels of the hybrid vertical coordinate"
# hyam,hybm see http://www.pa.op.dlr.de/~MattiaRighi/ncl/NCL_exercises.pdf
#              "The hybrid vertical representation"
# Pressure at level k p(k) = A(k) + B(k)Psurf
# I.e. coefficients for linear scaling of pressure at height based on surface pressure
import scipy.integrate as sin
def bmse_calc(T,qv, p): #,P0,PS,hyam,hybm):
    eps = 0.622 # Ratio of molecular weight(H2O)/molecular weight(dry air)
    R_D = 287 # Specific gas constant of dry air in J/K/kg
    Rv = 461
    # Calculate kappa
    QSAT0 = qsat(T,p) #P0,PS,hyam,hybm)
    kappa = 1+(L_V**2)*QSAT0/(Rv*C_P*(T**2))
    # Calculate geopotential
    r = qv/(qv**0-qv)
    Tv = T*(r**0+r/eps)/(r**0+r)
    #p = P0 * hyam + PS[:, None] * hybm
    p = p.astype(np.float32)
    RHO = p/(R_D*Tv)
    Z = -sin.cumulative_trapezoid(x=p,y=1/(G*RHO),axis=1)
    Z = np.concatenate((0*Z[:,0:1]**0,Z),axis=1)
    # Assuming near-surface is at 2 meters
    num_levels = T.shape[1]
    Z = (Z-Z[:,[num_levels - 1]])+2 
    # Calculate MSEs of plume and environment
    Tile_dim = [1,num_levels]
    h_plume = np.tile(np.expand_dims(C_P*T[:,-1]+L_V*qv[:,-1],axis=1),Tile_dim)
    h_satenv = C_P*T+L_V*qv+G*Z
    return (G/kappa)*(h_plume-h_satenv)/(C_P*T)



def add_vector_features(vector_dict):
    R_air = 287.0 # Mass-based gas constant approx for air in J/kg.K

    temperature_np = vector_dict['state_t']
    (rows, cols) = temperature_np.shape # use as template
    assert cols == num_atm_levels

    # Using fixed pressure levels, hopefully near enough, not sure in dataset whether
    # we're supposed to scale with surface pressure or something:
    pressure_np = np.tile(level_pressure_pa_np, (rows,1))
    vector_dict['pressure'] = pressure_np
    # pV = mRT
    # m/V = p/RT = density, with *100 for hPa -> Pa conversion
    density_np = pressure_np / (R_air * temperature_np)
    vector_dict['density'] = density_np
    recip_density_np = np.maximum(1.0 / density_np, 100.0) # Density gets v small so avoid dependent features blowing up
    vector_dict['recip_density'] = recip_density_np

    # Momentum per unit vol just density * velocity
    vel_zonal_np = vector_dict['state_u']
    vel_meridional_np = vector_dict['state_v']
    abs_wind_np = np.sqrt((vel_zonal_np ** 2) + (vel_meridional_np ** 2))
    vector_dict['abs_wind'] = abs_wind_np
    abs_momentum_np = density_np * abs_wind_np
    vector_dict['abs_momentum'] = abs_momentum_np
    zonal_stress_np = vector_dict['pbuf_TAUX']
    meridional_stress_np = vector_dict['pbuf_TAUY']
    vector_dict['abs_stress'] = np.sqrt((zonal_stress_np ** 2) + (meridional_stress_np ** 2))
    specific_humidity_np = vector_dict['state_q0001']
    rel_humidity_np = RH_from_climate(specific_humidity_np, temperature_np, pressure_np)
    vector_dict['rel_humidity'] = rel_humidity_np
    wind_rh_prod_np = rel_humidity_np * abs_wind_np
    vector_dict['wind_rh_prod'] = wind_rh_prod_np
    recip_rel_humidity_np = 1.0 / np.maximum(rel_humidity_np, 0.1)
    vector_dict['recip_rel_humidity'] = recip_rel_humidity_np
    buoyancy_np = bmse_calc(temperature_np, specific_humidity_np, pressure_np)
    vector_dict['buoyancy'] = buoyancy_np
    water_cloud_np = vector_dict['state_q0002']
    ice_cloud_np = vector_dict['state_q0003']
    tot_cloud_np = water_cloud_np + ice_cloud_np
    vector_dict['total_cloud'] = tot_cloud_np
    tot_cloud_density_np = tot_cloud_np * density_np
    vector_dict['total_cloud_density'] = tot_cloud_density_np
    vector_dict['recip_water_cloud'] = 1.0 / np.maximum(water_cloud_np, 1e-7)
    vector_dict['recip_ice_cloud'] = 1.0 / np.maximum(ice_cloud_np, 1e-7)
    wind_cloud_prod_np = tot_cloud_np * abs_wind_np
    vector_dict['wind_cloud_prod'] = wind_cloud_prod_np
    down_integ_tot_cloud_np = np.cumsum(tot_cloud_density_np) # lower indices higher altitude
    up_integ_tot_cloud_np = np.cumsum(tot_cloud_density_np[::-1])[::-1] # reverse before sum, then re-reverse
    vector_dict['down_integ_tot_cloud'] = down_integ_tot_cloud_np
    vector_dict['up_integ_tot_cloud'] = up_integ_tot_cloud_np
    # Some guesses here hard to find instantaneous values:
    total_gwp_np = vector_dict['pbuf_ozone'] * 1000.0 + vector_dict['pbuf_CH4'] * 200.0 + vector_dict['pbuf_N2O'] * 273.0
    vector_dict['total_gwp'] = total_gwp_np
    sensible_heat_flux_np = vector_dict['pbuf_SHFLX']
    latent_heat_flux_np = vector_dict['pbuf_LHFLX']
    longwave_surface_flux_np = vector_dict['cam_in_LWUP']
    # IR-absorbing gases are mixing ratio so density already factored in
    # (low density --> low GWP heat deposition --> same temperature rise as if dense)
    vector_dict['sensible_flux_gwp_prod'] = total_gwp_np * sensible_heat_flux_np
    vector_dict['up_lw_flux_gwp_prod'] = total_gwp_np * longwave_surface_flux_np
    # Assuming heating effect in K/s inversely proportional to heat capacity, i.e. density
    vector_dict['lat_heat_div_density'] = latent_heat_flux_np * recip_density_np
    vector_dict['sen_heat_div_density'] = sensible_heat_flux_np * recip_density_np

    # Single-value new features

    # Solar insolation adjusted for zenith angle (angle to vertical)
    vert_insolation_np = vector_dict['pbuf_SOLIN'] * vector_dict['pbuf_COSZRS']
    vector_dict['vert_insolation'] = vert_insolation_np
    # Absorbance of solar shortwave radiation
    direct_sw_absorb_np = vector_dict['vert_insolation'] * (1.0 - vector_dict['cam_in_ASDIR'])
    vector_dict['direct_sw_absorb'] = direct_sw_absorb_np
    diffuse_sw_absorb_np = vector_dict['vert_insolation'] * (1.0 - vector_dict['cam_in_ASDIF'])
    vector_dict['diffuse_sw_absorb'] = diffuse_sw_absorb_np
    # Absorbance of IR radiation from ground
    direct_lw_absorb_np = vector_dict['cam_in_LWUP'] * (1.0 - vector_dict['cam_in_ALDIR'])
    vector_dict['direct_lw_absorb'] = direct_lw_absorb_np
    diffuse_lw_absorb_np = vector_dict['cam_in_LWUP'] * (1.0 - vector_dict['cam_in_ALDIF'])
    vector_dict['diffuse_lw_absorb'] = diffuse_lw_absorb_np


re_vector_heading = re.compile('([A-Za-z0-9_]+?)_([0-9]+)')

def vectorise_data(pl_df, col_data):
    vector_dict = {}
    col_idx = 0
    while col_idx < len(pl_df.columns):
        col_name = pl_df.columns[col_idx]
        if not col_name or col_name == 'sample_id':
            col_idx += 1
            continue
        matcher = re_vector_heading.match(col_name)
        if matcher:
            # Element in vector column
            col_name = matcher.group(1) # now without indexing
            idx_str = matcher.group(2)
            idx = int(idx_str)
            if idx != 0:
                print(f"Error: expected zeroth element first, got {idx}")
                sys.exit(1)
            df_level_subset = pl_df[: , col_idx : col_idx + num_atm_levels]
            row_level_matrix = df_level_subset.to_numpy().astype(np.float64)
            col_idx += num_atm_levels
        else:
            # Scalar column
            row_vector = pl_df[: , col_name].to_numpy().astype(np.float64)
            row_vector = row_vector.reshape(len(row_vector), 1)
            col_info = col_data.unexpanded_cols_by_name.get(col_name)
            if col_info and not col_info.is_input and col_info.dimension <= 1:
                # Leave output scalars as they are
                row_level_matrix = row_vector
            else:
               row_level_matrix = np.tile(row_vector, (1,num_atm_levels))
            col_idx += 1
        vector_dict[col_name] = row_level_matrix

    return vector_dict

def mean_vector_across_samples(sample_list):
    """Given series of sample row vectors across data columns, form
    new row vector which is mean of those samples"""

    # join series of row vectors across inputs to one matrix of row samples vs column inputs
    all_samples = np.stack(sample_list)
    # Get mean down each column to get new row vector across input columns
    mean_vector = all_samples.mean(axis=0)
    return mean_vector

def minmax_vector_across_samples(sample_tuple_list, is_min):
    all_samples = np.stack(sample_tuple_list)
    if is_min:
        return np.min(all_samples, axis=0)
    else:
        return np.max(all_samples, axis=0)


def preprocess_data(pl_df, has_outputs, col_data, scaling_data, submission_weights_current, submission_weights_old):
    vector_dict = vectorise_data(pl_df, col_data)
    add_vector_features(vector_dict)
    # Glue input columns together by rows in batch, then feature, then atm level
    # (Works with torch.nn.Conv1d)
    for i, col_name in enumerate(col_data.unexpanded_input_col_names):
        if i == 0:
            x = vector_dict[col_name]
            (rows,cols) = x.shape
            x = x.reshape(rows, 1, cols)
        else:
            x = np.concatenate((x, vector_dict[col_name].reshape(rows, 1, cols)), axis=1)

    if has_outputs:
        # Leaving y-data expanded, model has to expand outputs to match
        for i, col_name in enumerate(col_data.unexpanded_output_col_names):
            if i == 0:
                y = vector_dict[col_name]
            else:
                y_add = vector_dict[col_name]
                y = np.concatenate((y, y_add), axis=1)
    else:
        y = None

    # Accumulate batch statistics to use in normalisation later from training data only
    if has_outputs:
        # Now applying same scaling across whole 60-level channel, for x at least:
        mx = x.mean(axis=(0,2))
        mx = mx.reshape(1, len(col_data.unexpanded_input_col_names), 1)
        scaling_data.mx_sample.append(mx)

        x_min = np.min(x, axis=(0,2))
        x_min = x_min.reshape(-1)
        scaling_data.xmin_sample.append(x_min)
        x_max = np.max(x, axis=(0,2))
        x_max = x_max.reshape(-1)
        scaling_data.xmax_sample.append(x_max)

        # Pre-scaling stats for understanding
        my_raw = y.mean(axis=0)
        scaling_data.my_sample_raw.append(my_raw)
        y_min = np.min(y, axis=0)
        y_max = np.max(y, axis=0)
        scaling_data.ymin_sample_raw.append(y_min)
        scaling_data.ymax_sample_raw.append(y_max)

        # Scaling outputs by old submission weights, so we get
        # rid of very tiny values that will give very small variances, and will make
        # them suitable hopefully for conversion to float32
        y = y * submission_weights_old

        # Also by current submission weights (though those are only 0 or 1 now) so we
        # have target values that match what is required for submission
        y = y * submission_weights_current

        my_weighted = y.mean(axis=0)
        scaling_data.my_sample_weighted.append(my_weighted)
        y_min = np.min(y, axis=0)
        y_max = np.max(y, axis=0)
        scaling_data.ymin_sample_weighted.append(y_min)
        scaling_data.ymax_sample_weighted.append(y_max)

    del pl_df
    gc.collect()

    return x, y


def normalise_data(x, y, scaling_data, has_outputs):

    # Original had mx.reshape(1,-1) to go from 1D row vector to 2D array with
    # one row but seems unnecessary
    x = (x - scaling_data.mx) / scaling_data.sx
    if not use_float64: x = x.astype(np.float32)

    if has_outputs:
        y = y / scaling_data.sy # Again experimenting with not using mean offset in y: (y - my) / sy
        if not use_float64: y = y.astype(np.float32)
    else:
        y = None

    return x, y

def load_submission_weights(col_data):
    # First row is all we need from sample submission, to get col weightings. 
    # sample_id column labels are identical to test.csv (checked first rows at least)
    print('Loading submission weights...')
    sample_submission_df = pl.read_csv(submission_template_path, n_rows=1)

    # Single row of weights for outputs
    submission_weights_current = sample_submission_df[col_data.expanded_names_output].to_numpy().astype(np.float64)

    # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/513193
    # Pre-18 June submission weights were good for getting values into sensible range for
    # float32 modelling; as string here so this works on Kaggle without input files available
    old_sample_submission_top_row = \
""",sample_id,ptend_t_0,ptend_t_1,ptend_t_2,ptend_t_3,ptend_t_4,ptend_t_5,ptend_t_6,ptend_t_7,ptend_t_8,ptend_t_9,ptend_t_10,ptend_t_11,ptend_t_12,ptend_t_13,ptend_t_14,ptend_t_15,ptend_t_16,ptend_t_17,ptend_t_18,ptend_t_19,ptend_t_20,ptend_t_21,ptend_t_22,ptend_t_23,ptend_t_24,ptend_t_25,ptend_t_26,ptend_t_27,ptend_t_28,ptend_t_29,ptend_t_30,ptend_t_31,ptend_t_32,ptend_t_33,ptend_t_34,ptend_t_35,ptend_t_36,ptend_t_37,ptend_t_38,ptend_t_39,ptend_t_40,ptend_t_41,ptend_t_42,ptend_t_43,ptend_t_44,ptend_t_45,ptend_t_46,ptend_t_47,ptend_t_48,ptend_t_49,ptend_t_50,ptend_t_51,ptend_t_52,ptend_t_53,ptend_t_54,ptend_t_55,ptend_t_56,ptend_t_57,ptend_t_58,ptend_t_59,ptend_q0001_0,ptend_q0001_1,ptend_q0001_2,ptend_q0001_3,ptend_q0001_4,ptend_q0001_5,ptend_q0001_6,ptend_q0001_7,ptend_q0001_8,ptend_q0001_9,ptend_q0001_10,ptend_q0001_11,ptend_q0001_12,ptend_q0001_13,ptend_q0001_14,ptend_q0001_15,ptend_q0001_16,ptend_q0001_17,ptend_q0001_18,ptend_q0001_19,ptend_q0001_20,ptend_q0001_21,ptend_q0001_22,ptend_q0001_23,ptend_q0001_24,ptend_q0001_25,ptend_q0001_26,ptend_q0001_27,ptend_q0001_28,ptend_q0001_29,ptend_q0001_30,ptend_q0001_31,ptend_q0001_32,ptend_q0001_33,ptend_q0001_34,ptend_q0001_35,ptend_q0001_36,ptend_q0001_37,ptend_q0001_38,ptend_q0001_39,ptend_q0001_40,ptend_q0001_41,ptend_q0001_42,ptend_q0001_43,ptend_q0001_44,ptend_q0001_45,ptend_q0001_46,ptend_q0001_47,ptend_q0001_48,ptend_q0001_49,ptend_q0001_50,ptend_q0001_51,ptend_q0001_52,ptend_q0001_53,ptend_q0001_54,ptend_q0001_55,ptend_q0001_56,ptend_q0001_57,ptend_q0001_58,ptend_q0001_59,ptend_q0002_0,ptend_q0002_1,ptend_q0002_2,ptend_q0002_3,ptend_q0002_4,ptend_q0002_5,ptend_q0002_6,ptend_q0002_7,ptend_q0002_8,ptend_q0002_9,ptend_q0002_10,ptend_q0002_11,ptend_q0002_12,ptend_q0002_13,ptend_q0002_14,ptend_q0002_15,ptend_q0002_16,ptend_q0002_17,ptend_q0002_18,ptend_q0002_19,ptend_q0002_20,ptend_q0002_21,ptend_q0002_22,ptend_q0002_23,ptend_q0002_24,ptend_q0002_25,ptend_q0002_26,ptend_q0002_27,ptend_q0002_28,ptend_q0002_29,ptend_q0002_30,ptend_q0002_31,ptend_q0002_32,ptend_q0002_33,ptend_q0002_34,ptend_q0002_35,ptend_q0002_36,ptend_q0002_37,ptend_q0002_38,ptend_q0002_39,ptend_q0002_40,ptend_q0002_41,ptend_q0002_42,ptend_q0002_43,ptend_q0002_44,ptend_q0002_45,ptend_q0002_46,ptend_q0002_47,ptend_q0002_48,ptend_q0002_49,ptend_q0002_50,ptend_q0002_51,ptend_q0002_52,ptend_q0002_53,ptend_q0002_54,ptend_q0002_55,ptend_q0002_56,ptend_q0002_57,ptend_q0002_58,ptend_q0002_59,ptend_q0003_0,ptend_q0003_1,ptend_q0003_2,ptend_q0003_3,ptend_q0003_4,ptend_q0003_5,ptend_q0003_6,ptend_q0003_7,ptend_q0003_8,ptend_q0003_9,ptend_q0003_10,ptend_q0003_11,ptend_q0003_12,ptend_q0003_13,ptend_q0003_14,ptend_q0003_15,ptend_q0003_16,ptend_q0003_17,ptend_q0003_18,ptend_q0003_19,ptend_q0003_20,ptend_q0003_21,ptend_q0003_22,ptend_q0003_23,ptend_q0003_24,ptend_q0003_25,ptend_q0003_26,ptend_q0003_27,ptend_q0003_28,ptend_q0003_29,ptend_q0003_30,ptend_q0003_31,ptend_q0003_32,ptend_q0003_33,ptend_q0003_34,ptend_q0003_35,ptend_q0003_36,ptend_q0003_37,ptend_q0003_38,ptend_q0003_39,ptend_q0003_40,ptend_q0003_41,ptend_q0003_42,ptend_q0003_43,ptend_q0003_44,ptend_q0003_45,ptend_q0003_46,ptend_q0003_47,ptend_q0003_48,ptend_q0003_49,ptend_q0003_50,ptend_q0003_51,ptend_q0003_52,ptend_q0003_53,ptend_q0003_54,ptend_q0003_55,ptend_q0003_56,ptend_q0003_57,ptend_q0003_58,ptend_q0003_59,ptend_u_0,ptend_u_1,ptend_u_2,ptend_u_3,ptend_u_4,ptend_u_5,ptend_u_6,ptend_u_7,ptend_u_8,ptend_u_9,ptend_u_10,ptend_u_11,ptend_u_12,ptend_u_13,ptend_u_14,ptend_u_15,ptend_u_16,ptend_u_17,ptend_u_18,ptend_u_19,ptend_u_20,ptend_u_21,ptend_u_22,ptend_u_23,ptend_u_24,ptend_u_25,ptend_u_26,ptend_u_27,ptend_u_28,ptend_u_29,ptend_u_30,ptend_u_31,ptend_u_32,ptend_u_33,ptend_u_34,ptend_u_35,ptend_u_36,ptend_u_37,ptend_u_38,ptend_u_39,ptend_u_40,ptend_u_41,ptend_u_42,ptend_u_43,ptend_u_44,ptend_u_45,ptend_u_46,ptend_u_47,ptend_u_48,ptend_u_49,ptend_u_50,ptend_u_51,ptend_u_52,ptend_u_53,ptend_u_54,ptend_u_55,ptend_u_56,ptend_u_57,ptend_u_58,ptend_u_59,ptend_v_0,ptend_v_1,ptend_v_2,ptend_v_3,ptend_v_4,ptend_v_5,ptend_v_6,ptend_v_7,ptend_v_8,ptend_v_9,ptend_v_10,ptend_v_11,ptend_v_12,ptend_v_13,ptend_v_14,ptend_v_15,ptend_v_16,ptend_v_17,ptend_v_18,ptend_v_19,ptend_v_20,ptend_v_21,ptend_v_22,ptend_v_23,ptend_v_24,ptend_v_25,ptend_v_26,ptend_v_27,ptend_v_28,ptend_v_29,ptend_v_30,ptend_v_31,ptend_v_32,ptend_v_33,ptend_v_34,ptend_v_35,ptend_v_36,ptend_v_37,ptend_v_38,ptend_v_39,ptend_v_40,ptend_v_41,ptend_v_42,ptend_v_43,ptend_v_44,ptend_v_45,ptend_v_46,ptend_v_47,ptend_v_48,ptend_v_49,ptend_v_50,ptend_v_51,ptend_v_52,ptend_v_53,ptend_v_54,ptend_v_55,ptend_v_56,ptend_v_57,ptend_v_58,ptend_v_59,cam_out_NETSW,cam_out_FLWDS,cam_out_PRECSC,cam_out_PRECC,cam_out_SOLS,cam_out_SOLL,cam_out_SOLSD,cam_out_SOLLD
0,test_169651,30981.265271661872,22502.432413914863,18894.14713004499,14514.244730542465,10944.348069459196,9065.01072024503,9663.669038687454,12688.557362943708,19890.17226527665,25831.37317235381,33890.367561807274,44122.94111025334,59811.25595068309,79434.07500078829,107358.80916894016,135720.8418348218,149399.8411114814,128492.95185325432,91746.23687305572,72748.76911097553,66531.53596840335,62932.30598423903,56610.26874314136,49473.14369220607,43029.18495420936,36912.67491908133,31486.93117928144,26898.072997215502,23316.638282978325,20459.73133196152,18385.68309639014,17111.405107656312,16337.80991958771,15857.759882318944,15580.902485189716,15497.59045982052,15612.2556996736,15797.88455410361,15974.218740897895,16130.395527176632,16261.310866446129,16371.892401608216,16397.019695140876,16325.463899570548,16228.641108112768,16191.809643436269,16341.207925934068,16645.711351490587,17005.493716683693,17430.29874509864,17907.24023203076,18431.55334008694,19032.471309392287,19701.355113141435,20408.236605392685,20967.20795006453,21194.427318009974,21088.521528526755,19437.91555757985,13677.902713248171,0,0,0,0,0,0,0,0,0,0,0,0,871528441401.8333,1083221770553.0684,147034752676.7702,35556045575.13566,35153369257.41337,46086368691.51654,24689305171.692936,11343276593.440475,5396624651.94418,2449353007.641508,1132225885.703891,579547849.1340877,330219246.7861086,207613930.3131764,144580292.27473342,109933282.92266414,88706603.092171,73819777.54163922,63615988.74519494,57250262.292053565,52976073.06761927,49653169.17819005,46544975.11484598,43167606.9599748,39724375.20499403,36317177.25886468,33057511.80930482,29869089.497658804,26982386.85583376,24416235.17215712,22273651.697369896,20553426.04804544,19216240.03357431,18167694.44812838,17501855.536957663,17169938.630597908,17005382.258644175,16998475.26752617,17082890.987979066,17227982.77516062,17445823.21630204,17757404.421785507,18346092.75160569,19400573.66632694,20506722.48296608,22469648.380506545,23432031.455169585,26204163.40545158,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,1000000000000000.0,3673829810926.31,371405570725.2526,14219163611.984406,3001863018.1934915,1432766589.9326108,884599805.0283787,560127980.1033351,386052567.7087711,287331851.051439,222703657.59538063,181069239.6264349,154620864.3164144,138093777.60284117,126605828.89875436,117967840.02553518,111005814.39518328,105186901.20678852,100168133.0295481,95568646.67416307,91457433.39515457,88871610.45308323,88829796.26374224,91398113.73291488,96585131.67000748,104507692.01463065,115895119.998433,131939701.08213414,154492946.00677127,183147918.17086875,215151374.22324687,247158314.6345976,266792879.42215955,279115128.29108113,370541510.87006927,0,0,0,0,0,0,0,0,0,0,0,0,877670509694.7871,1174826943136.8308,1270605570069.038,21727315470.5208,3159456646.5437946,1090653401.282219,727967089.8459107,384399548.9506704,290787296.9451616,232703218.45048887,197467462.7577736,174310890.8025987,160536437.73297343,153567098.77048483,152120124.9453068,153115566.6756177,153955545.42558223,153734675.21565756,154798666.36905554,163346213.58113608,180013139.3707387,200324358.8534948,220754613.1646765,241290935.478592,262868932.2066308,284448910.01847774,305681084.4142859,327605088.8575117,350473296.7263526,373964594.1196182,398396925.8173239,423528355.65716046,450447055.544388,478857006.4973163,508200335.7126168,537309657.5789208,566854568.2904652,594618842.9455439,619715928.2391286,641395460.8414665,663290039.7810476,689274894.631561,718208866.3397261,743951200.8024124,761776104.2945968,772911224.3082078,804001144.8046833,772448774.7758856,0,0,0,0,0,0,0,0,0,0,0,0,4613823.568205323,1999308.9343799097,904636.2296014762,433823.6123842511,207201.39055371704,107836.09164720173,57647.915219220784,40606.52305039815,47739.86647922776,51669.35493930698,56438.19768395407,60447.45665200092,65251.4153955275,71920.88588011517,78529.58115204438,83422.30217897324,87036.98552475807,90389.72631774022,93982.39165674087,97578.0099352472,101428.21366062944,104630.69200130588,105685.04322626138,103962.58423268417,99650.31670632094,94290.49986206587,89514.90144353417,85905.45713126978,82784.9857650212,79152.28707014346,74847.81017353121,70378.81859610273,65420.04643792357,59953.75184604176,54764.28281143022,50362.51288353384,46212.571031725325,41997.52779088816,37692.05148110484,33834.73460995647,31846.09764364542,31934.145655397457,31454.81247448105,30105.4073072481,26957.830283611693,27760.04479210889,29853.374336459365,19133.428743715107,0,0,0,0,0,0,0,0,0,0,0,0,7619940.584531054,3148394.472742347,1308415.0022178134,540515.7720745018,215237.1053603881,102546.7276372816,68453.67122640925,50692.59053608593,51487.52043139844,52104.76838400132,54019.39151917722,55856.02168787862,60347.30240270209,68990.96019017675,79096.88768563846,87574.33453690328,94158.56052476274,101903.63670531697,111746.9753834774,122460.65399236557,132086.69387474353,141041.48571028374,146354.09441287292,145953.09590059065,139496.8007888401,128508.85108217449,116665.51769667884,107458.39706309135,100259.97236694951,94108.98505029618,88439.89456238014,82734.9027659809,77061.08621371102,71333.5319243128,65999.72532130677,61798.9972058361,58237.356419617165,54715.10266341248,50825.84431702935,46059.17688689915,40740.26050401376,36335.80228304863,33981.57568605091,33589.7143390849,33988.88524112733,36272.9364507092,41183.34413717943,29194.12369278645,0.0040536134869726,0.0138824238058072,135129884.5084534,12219717.5342461,0.0090705273332672,0.0085898851680217,0.0215368188774867,0.0336321308942602"""
    fd = io.StringIO(old_sample_submission_top_row)

    old_submission_df = pl.read_csv(fd)
    submission_weights_old = old_submission_df[col_data.expanded_names_output].to_numpy().astype(np.float64)

    # "EDIT 2: As @churkinnikita points out, ptend_q0002 12-14 are no longer zeroed out"
    ptend_q0002_base_idx = col_data.output_expanded_first_idx_by_name["ptend_q0002"]
    submission_weights_old[0, ptend_q0002_base_idx + 12 : ptend_q0002_base_idx + 15] = \
                                              submission_weights_old[0, ptend_q0002_base_idx + 15]
    
    # As the new submission weights will zero out unwanted columns anyway, but we'll need to
    # divide by the old weights, turn zeroes to ones to avoid div by zero later
    submission_weights_old[submission_weights_old == 0.0] = 1.0

    return submission_weights_current, submission_weights_old


def preprocess_training_data(train_hf, num_train_rows, col_data, submission_weights_current, submission_weights_old):
    # Preprocess entire dataset, gathering statistics for subsequent normalisation along the way

    print("No previous scalings so starting afresh")
    scaling_data = ScalingMetadata()
    scaling_data.mx_sample = [] # Each element vector of means of input columns, from one holo batch
    scaling_data.sx_sample = [] # ... and scaling factor
    scaling_data.my_sample_raw = []
    scaling_data.my_sample_weighted = []
    scaling_data.sy_sample = []
    scaling_data.xmin_sample = []
    scaling_data.xmax_sample = []
    scaling_data.ymin_sample_raw = []
    scaling_data.ymax_sample_raw = []
    scaling_data.ymin_sample_weighted = []
    scaling_data.ymax_sample_weighted = []

    for true_file_index in range(subset_base_row, subset_base_row + num_train_rows, cache_batch_size):
                        # Process slice of large dataframe corresponding to batch
        prenorm_cache_filename = f'{true_file_index}_prenorm.pkl'
        prenorm_cache_path = os.path.join(batch_cache_dir, prenorm_cache_filename)
        postnorm_cache_filename = f'{true_file_index}.pkl'
        postnorm_cache_path = os.path.join(batch_cache_dir, postnorm_cache_filename)

        if not os.path.exists(postnorm_cache_path) and not os.path.exists(prenorm_cache_path):
            print(f"Building {prenorm_cache_path}...")
            pl_slice_df = train_hf.get_slice(true_file_index, true_file_index + cache_batch_size)
            cache_np_x, cache_np_y = preprocess_data(pl_slice_df, True, col_data, scaling_data, submission_weights_current, submission_weights_old)
            if show_timings: print(f'HoloDataset slice build at row {true_file_index} took {time.time() - start_time} s')
            start_time = time.time()
            with open(prenorm_cache_path, 'wb') as fd:
                pickle.dump((cache_np_x, cache_np_y), fd)
                fd.flush() # Attempting to avoid random Ubuntu resets
                os.fsync(fd)
            del cache_np_x, cache_np_y
            gc.collect()

    # Mean of means gives us overall mean for each quantity (whole vectors for inputs,
    # individual columns for outputs with their differing submission weightings)
    scaling_data.mx = mean_vector_across_samples(scaling_data.mx_sample)
    scaling_data.my_raw = mean_vector_across_samples(scaling_data.my_sample_raw)
    scaling_data.my_weighted = mean_vector_across_samples(scaling_data.my_sample_weighted)

    # Range limits:
    scaling_data.xmin = minmax_vector_across_samples(scaling_data.xmin_sample, True)
    scaling_data.xmax = minmax_vector_across_samples(scaling_data.xmax_sample, False)
    scaling_data.ymin_raw = minmax_vector_across_samples(scaling_data.ymin_sample_raw, True)
    scaling_data.ymax_raw = minmax_vector_across_samples(scaling_data.ymax_sample_raw, False)
    scaling_data.ymin_weighted = minmax_vector_across_samples(scaling_data.ymin_sample_weighted, True)
    scaling_data.ymax_weighted = minmax_vector_across_samples(scaling_data.ymax_sample_weighted, False)

    # Do another pass to get standard deviation stats across whole dataset, which we
    # needed means across whole dataset for.
    # Need this for R2 metric even using range scaling instead of stdev
    x_sumsq_sample = []
    y_sumsq_sample = []
    for true_file_index in range(subset_base_row, subset_base_row + num_train_rows, cache_batch_size):
        prenorm_cache_filename = f'{true_file_index}_prenorm.pkl'
        prenorm_cache_path = os.path.join(batch_cache_dir, prenorm_cache_filename)
        postnorm_cache_filename = f'{true_file_index}.pkl'
        postnorm_cache_path = os.path.join(batch_cache_dir, postnorm_cache_filename)

        if not os.path.exists(postnorm_cache_path):
            print(f"Getting variances from {prenorm_cache_path}...")
            with open(prenorm_cache_path, 'rb') as fd:
                (x_prenorm, y_prenorm) = pickle.load(fd)
            x_diffs_sqd = (x_prenorm - scaling_data.mx) ** 2
            x_sum_sqs = x_diffs_sqd.sum(axis=(0,2)) # sum over batch rows and over atm layers to leave num vector features
            x_sumsq_sample.append(x_sum_sqs)
            y_diffs_sqd = (y_prenorm - scaling_data.my_weighted) ** 2
            y_sum_sqs = y_diffs_sqd.sum(axis=0)
            y_sumsq_sample.append(y_sum_sqs)
            del x_prenorm, y_prenorm, x_diffs_sqd, y_diffs_sqd
            gc.collect() # Had a couple of spontaneous Ubuntu reboots here previously

        # Now normalise whole dataset using statistics gathered during preprocessing
        # Using scaling found in training data; though could use test data if big enough?

        x_sumsq_avg = mean_vector_across_samples(x_sumsq_sample) / (cache_batch_size * num_atm_levels)
        y_sumsq_avg = mean_vector_across_samples(y_sumsq_sample) / cache_batch_size
        sx = np.sqrt(x_sumsq_avg)
        scaling_data.stdev_x = sx.reshape((1,len(col_data.unexpanded_input_col_names),1))
        scaling_data.stdev_y = np.maximum(np.sqrt(y_sumsq_avg), min_std)

        if scale_using_range_limits:
            sx = (scaling_data.xmax - scaling_data.xmin) / 2.0 # aiming for [-1, 1] normalised range
            scaling_data.sx = np.maximum(sx, min_std).reshape(scaling_data.mx.shape)
            bigger_y_lim = np.maximum(np.abs(scaling_data.ymin_weighted), np.abs(scaling_data.ymax_weighted)) # as not centring with mean
            scaling_data.sy = np.maximum(bigger_y_lim, min_std)
        else:
            scaling_data.sx = scaling_data.stdev_x
            # Donor notebook used RMS instead of stdev here, discussion thread suggesting that
            # gives loss value like competition criterion but I see no training advantage:
            # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/498806
            scaling_data.sy = scaling_data.stdev_y

    for true_file_index in range(subset_base_row, subset_base_row + num_train_rows, cache_batch_size):
                        # Process slice of large dataframe corresponding to batch
        prenorm_cache_filename = f'{true_file_index}_prenorm.pkl'
        postnorm_cache_filename = f'{true_file_index}.pkl'
        
        prenorm_cache_path = os.path.join(batch_cache_dir, prenorm_cache_filename)
        postnorm_cache_path = os.path.join(batch_cache_dir, postnorm_cache_filename)

        if not os.path.exists(postnorm_cache_path):
            print(f"Building {postnorm_cache_path}...")
            with open(prenorm_cache_path, 'rb') as fd:
                (x_prenorm, y_prenorm) = pickle.load(fd)

            x_postnorm, y_postnorm = normalise_data(x_prenorm, y_prenorm,
                                                    scaling_data, True)
            x_raw_trick_features = extract_raw_trick_subset(x_prenorm, col_data)
            if show_timings: print(f'HoloDataset slice build at row {true_file_index} took {time.time() - start_time} s')
            start_time = time.time()
            with open(postnorm_cache_path, 'wb') as fd:
                pickle.dump((x_postnorm, y_postnorm, x_raw_trick_features), fd)
                fd.flush()
                os.fsync(fd)
            os.remove(prenorm_cache_path)
            del x_postnorm, y_postnorm
            gc.collect()

    # Save scalings, need them to process test data if do a rerun
    with open(scaling_cache_path, 'wb') as fd:
        del scaling_data.mx_sample
        del scaling_data.my_sample_raw
        del scaling_data.my_sample_weighted
        del scaling_data.xmin_sample
        del scaling_data.xmax_sample
        del scaling_data.ymin_sample_raw
        del scaling_data.ymax_sample_raw
        del scaling_data.ymin_sample_weighted
        del scaling_data.ymax_sample_weighted
        pickle.dump(scaling_data, fd)
        print("Saved scalings for next time")

    return scaling_data


def extract_raw_trick_subset(x_prenorm, col_data):
    # Cloud tendency trick needs original float64 precision data for relevant features only
    trick_idx = []
    for col_name in col_data.input_trick_names:
        x_idx = col_data.feature_idx_by_name[col_name]
        trick_idx.append(x_idx)
    raw_trick_features = x_prenorm[:, trick_idx, :]
    return raw_trick_features


def write_scaling_stats_to_file(scaling_data, col_data):
    x_df = pl.DataFrame()
    x_df = x_df.with_columns(pl.Series(name="var", values=col_data.unexpanded_input_col_names))
    x_df = x_df.with_columns(pl.from_numpy(np.reshape(scaling_data.xmin, (-1)), ["raw min"]))
    x_df = x_df.with_columns(pl.from_numpy(np.reshape(scaling_data.xmax, (-1)), ["raw max"]))
    x_df = x_df.with_columns(pl.from_numpy(np.reshape(scaling_data.mx, (-1)), ["mean"]))
    x_df = x_df.with_columns(pl.from_numpy(np.reshape(scaling_data.sx, (-1)), ["scale"]))
    y_df = pl.DataFrame()
    y_df = y_df.with_columns(pl.Series(name="var", values=col_data.expanded_names_output))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.ymin_raw, ["raw min"]))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.ymax_raw, ["raw max"]))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.ymin_weighted, ["weighted min"]))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.ymax_weighted, ["weighted max"]))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.my_weighted, ["mean"]))
    y_df = y_df.with_columns(pl.from_numpy(scaling_data.sy, ["scale"]))
    tot_df = pl.concat([x_df,y_df], how="diagonal")
    tot_df.write_csv("scaling_stats.csv")


class AtmLayerCNN(nn.Module):
    def __init__(self, col_data, gen_conv_width=7, gen_conv_depth=15, init_1x1=True, 
                 norm_type="layer", activation_type="silu", poly_degree=0):
        super().__init__()
        
        if use_float64:
            dtype = torch.float64
        else:
            dtype = torch.float32

        self.col_data = col_data
        self.init_1x1 = init_1x1
        self.norm_type = norm_type
        self.activation_type = activation_type
        self.poly_degree = poly_degree

        num_input_feature_chans = len(col_data.cnn_input_feature_idx)
        if do_feature_knockout:
            num_input_feature_chans -= 1

        # Initialize the layers

        input_size = num_input_feature_chans

        if self.init_1x1:
            # Having single-width layer just means mixing combinations of input
            # vars for this layer only as a useful starting point
            # Could generalise to multiple if works
            output_size = num_input_feature_chans * gen_conv_depth
            self.conv_layer_1x1 = nn.Conv1d(input_size, output_size, 1,
                                    padding='same', dtype=dtype)
            self.layernorm_1x1 = self.norm_layer(output_size, num_atm_levels, dtype=dtype)
            self.activation_layer_1x1 = self.activation_layer(dtype=dtype)
            self.dropout_layer_1x1 = nn.Dropout(p=dropout_p)
        else:
            # No initial unit width layer
            output_size = input_size

        input_size = output_size
        output_size = num_input_feature_chans * gen_conv_depth
        self.conv_layer_0 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same', dtype=dtype)
        self.layernorm_0 = self.norm_layer(output_size, num_atm_levels, dtype=dtype)
        self.activation_layer_0 = self.activation_layer(dtype=dtype)
        self.dropout_layer_0 = nn.Dropout(p=dropout_p)

        input_size = output_size
        output_size = num_input_feature_chans * gen_conv_depth
        self.conv_layer_1 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same', dtype=dtype)
        self.layernorm_1 = self.norm_layer(output_size, num_atm_levels, dtype=dtype)
        self.activation_layer_1 = self.activation_layer(dtype=dtype)
        self.dropout_layer_1 = nn.Dropout(p=dropout_p)

        input_size = output_size
        self.last_conv_depth = gen_conv_depth
        output_size = col_data.num_all_outputs_as_vectors * self.last_conv_depth
        self.conv_layer_2 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same', dtype=dtype)
        self.layernorm_2 = self.norm_layer(output_size, num_atm_levels, dtype=dtype)
        self.activation_layer_2 = self.activation_layer(dtype=dtype)
        self.dropout_layer_2 = nn.Dropout(p=dropout_p)

        # Output layer - no dropout, no activation function
        # Data is structured such that all vector columns come first, then scalars
        input_size = output_size
        self.conv_vector_harvest = nn.Conv1d(col_data.num_pure_vector_outputs * gen_conv_depth,
                                         col_data.num_pure_vector_outputs, 1, padding='same', dtype=dtype)
        self.vector_flatten = nn.Flatten()
        self.pre_scalar_pool = nn.AvgPool1d(num_atm_levels)
        self.scalar_flatten = nn.Flatten()
        self.linear_scalar_harvest = nn.Linear(col_data.num_scalar_outputs * gen_conv_depth, 
                                               col_data.num_scalar_outputs, dtype=dtype)

        # Polynomial on top of that to hopefully cope better with wide dynamic
        # range of some outputs, initialising close to straight-through linear
        # though
        # https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html
        # Cubic probably a bit ambitious?
        # No 'a' offset because last layer will have own learnt bias
        self.vector_poly_coeffs = self.create_coeff_param_list((col_data.num_pure_vector_outputs,1))
        self.scalar_poly_coeffs = self.create_coeff_param_list((col_data.num_scalar_outputs))

    def create_coeff_param_list(self, vector_shape):
        coeffs = []
        for i in range(2, self.poly_degree + 1):
            coeffs_this_degree = nn.Parameter(torch.randn(vector_shape, dtype=torch.float32))
            coeffs.append(coeffs_this_degree)
        return nn.ParameterList(coeffs) # to register properly as parameters

    def norm_layer(self, num_features, num_chans, dtype=torch.float32):
            match self.norm_type:
                case "layer":
                    return nn.LayerNorm([num_features, num_chans], dtype=dtype)
                case "batch":
                    return nn.BatchNorm1d(num_features, dtype=dtype)
                case _:
                    print(f"Unexpected norm_type={self.norm_type}")
                    sys.exit(1)

    def activation_layer(self, dtype=torch.float32):
            match self.activation_type:
                case "silu":
                    return nn.SiLU() # Now skipping inplace=True
                case "prelu":
                    return nn.PReLU()
                case _:
                    print(f"Unexpected activation_type={self.activation_type}")
                    sys.exit(1)


    def forward(self, x):
        if self.init_1x1:
            x = self.conv_layer_1x1(x)
            x = self.layernorm_1x1(x)
            x = self.activation_layer_1x1(x)
            x = self.dropout_layer_1x1(x)
        x = self.conv_layer_0(x)
        x = self.layernorm_0(x)
        x = self.activation_layer_0(x)
        x = self.dropout_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.layernorm_1(x)
        x = self.activation_layer_1(x)
        x = self.dropout_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.layernorm_2(x)
        x = self.activation_layer_2(x)
        x = self.dropout_layer_2(x)
        num_conv_outputs_for_vectors = self.col_data.num_pure_vector_outputs * self.last_conv_depth
        vector_subset = x[:, : num_conv_outputs_for_vectors, :]
        scalar_subset = x[:, num_conv_outputs_for_vectors :, :]
        scalars_pooled = self.pre_scalar_pool(scalar_subset)
        scalars_flattened = self.scalar_flatten(scalars_pooled)
        scalar_harvest = self.linear_scalar_harvest(scalars_flattened)
        vector_harvest = self.conv_vector_harvest(vector_subset)
        # Polynomial output an attempt to deal with wide-ranging outlier values
        if (self.poly_degree < 2):
            scalars_polynomial = scalar_harvest
            vectors_polynomial = vector_harvest
        elif (self.poly_degree == 2):
            scalars_polynomial = (scalar_harvest + 
                                  (self.scalar_poly_coeffs[0] * 0.05) * scalar_harvest ** 2)
            vectors_polynomial = (vector_harvest + 
                                  (self.vector_poly_coeffs[0] * 0.05) * vector_harvest ** 2)
        elif (self.poly_degree == 3):
            scalars_polynomial = (scalar_harvest + 
                                  (self.scalar_poly_coeffs[0] * 0.05 ) * scalar_harvest ** 2 +
                                  (self.scalar_poly_coeffs[1] * 0.005) * scalar_harvest ** 3)
            vectors_polynomial = (vector_harvest + 
                                  (self.vector_poly_coeffs[0] * 0.05 ) * vector_harvest ** 2 +
                                  (self.vector_poly_coeffs[1] * 0.005) * vector_harvest ** 3)
        else:
            sys.exit(1)
        vectors_flattened = self.vector_flatten(vectors_polynomial)
        expanded_outputs = torch.cat((vectors_flattened, scalars_polynomial), dim=1)
        return expanded_outputs
#

class HoloDataset(Dataset):
    def __init__(self, holo_df, cache_rows, exec_data, device, x_col_idx_list):
        """
        Initialize with HoloFrame instance.
        """
        self.holo_df = holo_df
        self.cache_rows = cache_rows
        self.exec_data = exec_data
        self.device = device
        self.x_col_idx_list = x_col_idx_list
        self.cache_base_idx = -1
        self.cache_np_x = None
        self.cache_np_y = None
        self.cache_np_trick_features = None

    def __len__(self):
        """
        Total number of samples.
        """
        return len(self.holo_df)

    def ensure_block_loaded(self, index):
        if not (self.cache_base_idx >= 0 and 
                index >= self.cache_base_idx and index < self.cache_base_idx + self.cache_rows):
            # We don't already have this in RAM
            start_time = time.time()
            self.cache_base_idx = index
            true_file_idx = self.cache_base_idx + subset_base_row
            cache_filename = f'{true_file_idx}.pkl'
            cache_path = os.path.join(batch_cache_dir, cache_filename)
            if os.path.exists(cache_path):
                # Preprocessing already done, retrieve from binary cache file
                with open(cache_path, 'rb') as fd:
                    (all_np_x, self.cache_np_y, self.cache_np_trick_features) = pickle.load(fd)
                if show_timings: print(f'HoloDataset slice cache load at row {self.cache_base_idx} took {time.time() - start_time} s')    
            else:
                print(f"ERROR: cached data not found at row {index}")
                sys.exit(1)
            # Unwanted features removed only when used here so different model types with
            # different feature subsets can share cached disk data with all features
            self.cache_np_x = all_np_x[:, self.x_col_idx_list, :]

    def __getitem__(self, index):
        """
        Generate one sample of data for PyTorch as tensors.
        """
        self.ensure_block_loaded(index)
        # Convert the RAM numpy data to tensors when requested
        cache_idx = index - self.cache_base_idx
        x_np = self.cache_np_x[cache_idx]
        y_np = self.cache_np_y[cache_idx]
        if do_feature_knockout:
            # Doing this post-processing so cached data can be used unchanged throughout
            x_np = np.delete(x_np, self.exec_data.feature_knockout_idx, axis=0)
        return torch.from_numpy(x_np).to(self.device), \
               torch.from_numpy(y_np).to(self.device)
    
    def get_np_block_slice(self, block_base_idx, end_idx):
        row_idx = block_base_idx if block_base_idx >= 0 else self.cache_base_idx
        complete_x = None
        complete_y = None
        complete_tricks = None
        while True:
            self.ensure_block_loaded(row_idx)
            if complete_x is None:
                complete_x = self.cache_np_x
                complete_y = self.cache_np_y
                complete_tricks = self.cache_np_trick_features
            else:
                complete_x = np.concatenate((complete_x, self.cache_np_x), axis=0)
                complete_y = np.concatenate((complete_y, self.cache_np_y), axis=0)
                complete_tricks = np.concatenate((complete_tricks, self.cache_np_trick_features), axis=0)
            row_idx += self.cache_rows
            if row_idx >= end_idx:
                break
        num_reqd_rows = end_idx - block_base_idx
        captured_rows = complete_x.shape[0] 
        if captured_rows > num_reqd_rows:
            complete_x = complete_x[:num_reqd_rows,:,:]
            complete_y = complete_y[:num_reqd_rows,:,:]
            complete_tricks = complete_tricks[:num_reqd_rows,:,:]
            
        return complete_x, complete_y, complete_tricks



def form_row_range_from_block_range(block_indices, num_train_rows, num_blocks, num_full_blocks, batch_size):
    blocks_divide_exactly = (num_blocks == num_full_blocks)
    row_idx = []
    for block_idx in block_indices:
        if block_idx == num_blocks - 1 and not blocks_divide_exactly:
            # Smaller number in last dangling block
            num_rows_in_block = num_train_rows - num_full_blocks * batch_size
        else:
            num_rows_in_block = batch_size
        base_row_idx = block_idx * batch_size
        row_idx.extend(list(range(base_row_idx, base_row_idx + num_rows_in_block)))
    return row_idx



def unscale_outputs(y, scaling_data, submission_weights_old):
    """Undo normalisation to return to true values (but with submission
    weights still multiplied in)"""

    # Return to float64 if not already to cope with tiny output values
    y = y.astype(np.float64)

    # Should now be safe to divide by old submission weights to get correct magnitudes
    # (New submission weights still present but 0 or 1, and in any case want those
    # to be part of final target values for submission)
    y = y / submission_weights_old

    # undo y scaling
    y = (y * scaling_data.sy) # Experimenting again with no mean offset: + my

    return y


def postprocess_predictions(x, y, trick_x, scaling_data, col_data):
    zeroed_cols = []
    for i in range(scaling_data.sy.shape[0]):
        # CW: still using original threshold although now premultiplying outputs by
        # submission weightings, though does zero out those with zero weights
        # (and some others)
        col_name = col_data.expanded_names_output[i]
        if col_name in col_data.zero_or_bad_cols_by_name:
            if col_data.zero_or_bad_cols_by_name[col_name]:
                # One we officially zero
                y[:,i] = 0.0
            else:
                # Not one we officially zero so go for mean instead
                y[:,i] = scaling_data.my_raw[i] # 0 here if restore addition of mean offset later

    # Trick to predict some difficult tiny-value columns discussed here:
    # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484
    # The gist of it is that the cloud columns at high altitudes often have zero
    # values, and if they are non-zero then there is a good chance that the next
    # point in time (1200 s or 20 min later) will be zero. So we can guess the
    # tendency per second assuming the next point would be zero, giving the -1200 division.
    # Looking at first 11 rows of training data, the relationships work well for:
    # ptend_q0002[i] = state_q0002[i] / -1200   i=[12..30]  cloud ice mixing ratio
    # ptend_q0003[i] = state_q0003[i] / -1200   i=[12..18]  cloud ice mixing ratio
    # After that it starts to break down, though mixing some proportion of this guess
    # into the predicted value might still help.
    replace_cloud_tendency_trick("state_q0002", "ptend_q0002", 12, 29, x, y, trick_x, scaling_data, col_data)
#    replace_cloud_tendency_trick("state_q0003", "ptend_q0003", 12, 17, x, y, trick_x, scaling_data, col_data)

    return y

def replace_cloud_tendency_trick(src_name, dest_name, first_idx, last_idx, x, y, trick_x, scaling_data, col_data):
    src_full_set_idx = col_data.input_trick_names.index(src_name)
    raw_src_f64 = trick_x[:, src_full_set_idx, first_idx : last_idx].reshape((x.shape[0], 1, -1))
    y_base_idx = col_data.output_expanded_first_idx_by_name[dest_name]
    y[:, y_base_idx + first_idx : y_base_idx + last_idx] = raw_src_f64.reshape((y.shape[0],-1)) / -1200.0

class AnalysisData():
    def __init__(self):
        self.df = pl.DataFrame()
        self.num_rows = 0
        self.r2_raw = 0.0
        self.r2_clean = 0.0
        self.r2_vec = None


def analyse_batch(analysis_data, inputs_np, outputs_pred_np, outputs_true_np, trick_x_np, col_data, 
                  scaling_data, submission_weights_old):
    """Analyse batch of true versus predicted outputs"""

    # Return to original output scalings (but with submission weights
    # multiplied in) to match competition metric
    outputs_pred_np = unscale_outputs(outputs_pred_np, scaling_data, submission_weights_old)
    outputs_true_np = unscale_outputs(outputs_true_np, scaling_data, submission_weights_old)
    # Post-model tweaks for tricky columns
    outputs_pred_np = postprocess_predictions(inputs_np, outputs_pred_np, trick_x_np, scaling_data, col_data)

    # Assuming variance of dataset outputs found in training more 
    # representative than variance in this small batch?
    true_variance_sqd_scaled = scaling_data.stdev_y ** 2 # undo sqrt in stdev
    # Those variances were after scaling with old weightings, undo to compare
    # with values scaled for test output
    true_variance_sqd_raw = true_variance_sqd_scaled / submission_weights_old ** 2

    error_residues = outputs_true_np - outputs_pred_np
    error_variance_sqd = np.square(error_residues)
    num_new_rows = error_residues.shape[0]
    new_tot_rows = analysis_data.num_rows + num_new_rows
    factor_prev = analysis_data.num_rows / new_tot_rows
    factor_new = 1.0 - factor_prev
    avg_error_variance_sqd = np.mean(error_variance_sqd, axis=0)
    r2_metric = 1.0 - (avg_error_variance_sqd / true_variance_sqd_raw)
    if analysis_data.num_rows <= 0:
        analysis_data.r2_vec = np.zeros_like(r2_metric)
    analysis_data.r2_vec = factor_prev * analysis_data.r2_vec + factor_new * r2_metric
    all_col_r2 = np.mean(r2_metric)
    analysis_data.r2_raw = factor_prev * analysis_data.r2_raw + factor_new * all_col_r2
    good_cols = r2_metric[np.where(r2_metric > 0.0)]
    good_col_r2_sum = np.sum(good_cols)
    avg_r2_zeroed_bad_cols = good_col_r2_sum / r2_metric.shape[1]
    analysis_data.r2_clean = factor_prev * analysis_data.r2_clean + factor_new * avg_r2_zeroed_bad_cols
    analysis_data.num_rows += num_new_rows
    print(f"Batch all R2={all_col_r2}, bad excl R2={avg_r2_zeroed_bad_cols}")

    if do_analysis and analysis_data.df.height < max_analysis_output_rows:
        r2_cols = np.tile(r2_metric, (num_new_rows,1))
        batch_df = pl.DataFrame()
        for i, output_name in enumerate(col_data.expanded_names_output):
            batch_df = batch_df.with_columns(pl.from_numpy(outputs_true_np[:,i], [output_name + "_true"]))
            batch_df = batch_df.with_columns(pl.from_numpy(outputs_pred_np[:,i], [output_name + "_pred"]))
            batch_df = batch_df.with_columns(pl.from_numpy(r2_cols[:,i], [output_name + "_r2"]))
        for i, vector_name in enumerate(col_data.unexpanded_output_vector_col_names):
            first_col_idx = i * num_atm_levels
            end_col_idx = first_col_idx + num_atm_levels
            r2_avg_for_vector = np.mean(r2_cols[:, first_col_idx:end_col_idx], axis=1)
            batch_df = batch_df.with_columns(pl.from_numpy(r2_avg_for_vector, [vector_name + "_vecavgr2"]))
        analysis_data.df = pl.concat([analysis_data.df, batch_df])
    

def calc_output_r2_ranking(col_data, analysis_data):
    """List columns in order of R2 goodness to identify worst offenders"""

    bad_r2_names = []
    ranking_list = []
    for i, col_name in enumerate(col_data.expanded_names_output):
        r2_name = col_name + "_r2"
        r2_avg = analysis_data.r2_vec[0, i].mean()
        if r2_avg <= 0.0:
            # Will use mean for this column instead as prediction worse than that
            bad_r2_names.append(col_name)
        r2_description = col_data.expanded_cols_by_name[col_name].description
        ranking_list.append((r2_name, r2_avg, r2_description))
    ranking_list.sort(key=lambda x: x[1])
    ranking_df = pl.DataFrame(ranking_list, ["Var name", "R2", "Description"])
    ranking_df.write_csv(r2_ranking_path)

    return bad_r2_names


def do_cnn_training(model_params, exec_data, col_data, scaling_data, submission_weights_old,
                    param_permutations, train_loader, val_loader, cnn_dataset, device):
    warnings.filterwarnings('ignore', category=FutureWarning)

    if is_rerun and os.path.exists(epoch_counter_path):
        try:
            with open(epoch_counter_path) as fd:
                epochs_str = fd.readline()
                tot_epochs = int(epochs_str)
        except:
            print(f"Failed to open/parse {epoch_counter_path}, zeroing counter")
            tot_epochs = 0
    else:
        tot_epochs = 0

    model = AtmLayerCNN(col_data, **model_params).to(device)
    if try_reload_model and os.path.exists(exec_data.model_save_path):
        print('Attempting to reload model from disk...')
        model.load_state_dict(torch.load(exec_data.model_save_path))

    best_val_loss = float('inf')  # Set initial best as infinity
    criterion = nn.MSELoss()  # Using MSE for regression
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    torch.autograd.set_detect_anomaly(debug)
    exec_data.best_model_state = None       # To store the best model's state
    patience_count = 0

    if len(param_permutations) > 1:
        tot_epochs = 0

    for epoch in range(max_epochs):
        if do_train:
            model.train()
            total_loss = 0
            for batch_idx, (inputs, outputs_true) in enumerate(train_loader):
                start_time = time.time()
                optimizer.zero_grad()
                outputs_pred = model(inputs)
                loss = criterion(outputs_pred, outputs_true)
                loss.backward() # Calculates gradients by backpropagation (chain rule)
                optimizer.step()

                total_loss += loss.item()

                if show_timings: print(f'Training batch of {cnn_batch_size} took {time.time() - start_time} s')

                # Print every n steps
                if (batch_idx + 1) % batch_report_interval == 0:
                    print(f'Epoch {tot_epochs + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / batch_report_interval:.4f}')
                    total_loss = 0  # Reset the loss for the next n steps
        
        # Validation step
        analysis_data = AnalysisData()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, outputs_true) in enumerate(val_loader):
                outputs_pred = model(inputs)
                val_loss += criterion(outputs_pred, outputs_true).item()
                if do_analysis:
                    outputs_pred_np = outputs_pred.cpu().numpy()
                    outputs_true_np = outputs_true.cpu().numpy()
                    _, _, trick_x = cnn_dataset.get_np_block_slice(-1, cnn_batch_size)
                    analyse_batch(analysis_data, inputs.cpu().numpy(), outputs_pred_np, outputs_true_np,
                                   trick_x, col_data, scaling_data, submission_weights_old)

                if (batch_idx + 1) % batch_report_interval == 0:
                    print(f'Validation batch {batch_idx + 1}')

        avg_val_loss = val_loss / len(val_loader)

        tot_epochs += 1
        with open(epoch_counter_path, 'w') as fd:
            fd.write(f'{tot_epochs}\n')

        print(f'Epoch {tot_epochs}, Validation Loss: {avg_val_loss}, R2: {analysis_data.r2_clean}')
        with open(loss_log_path, 'a') as fd:
            fd.write(f'{tot_epochs},{avg_val_loss},{analysis_data.r2_clean}\n')

        
        scheduler.step(avg_val_loss)  # Adjust learning rate

        if do_analysis:
            analysis_data.df.head(max_analysis_output_rows).write_csv(analysis_df_path)
            bad_r2_output_names = calc_output_r2_ranking(col_data, analysis_data)

        if avg_val_loss < exec_data.overall_best_val_metric:
            exec_data.overall_best_val_metric = avg_val_loss
            exec_data.overall_best_model = model
            exec_data.overall_best_model_state = model.state_dict() # TODO is this static anyway?
            exec_data.overall_best_model_name = exec_data.model_save_path
            exec_data.best_feature_knockout_idx = exec_data.feature_knockout_idx
            print(f"{exec_data.model_save_path} best so far")

        # Update best model if current epoch's validation loss is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save the best model state
            patience_count = 0
            print("Validation loss decreased, saving new best model and resetting patience counter.")
            torch.save(model.state_dict(), exec_data.model_save_path)
        else:
            patience_count += 1
            print(f"No improvement in validation loss for {patience_count} epochs.")

        gc.collect()

        if patience_count >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

        if os.path.exists(stopfile_path):
            print("Stop file detected, deleting it and stopping now")
            os.remove(stopfile_path)
            exec_data.stop_requested = True
            break

    with open(cnn_analysis_data_path, 'wb') as fd:
        pickle.dump(analysis_data, fd)

    return bad_r2_output_names


def do_catboost_training(exec_data, col_data, scaling_data, submission_weights_old, dataset, train_block_idx,
                         val_block_idx,
                         iterations=400, depth=8, learning_rate=0.25,
                         border_count=32, l2_leaf_reg=5):
    # Catboost, mutually exclusive to start with

    cat_params = {
                    'iterations': iterations, 
                    'depth': depth, 
                    'task_type' : "CPU" if machine == "narg" else "GPU",
                    'use_best_model': False, # requires validation data
                    #'eval_metric': 'R2',
                    'loss_function': 'MultiRMSE',
                    'early_stopping_rounds': 200,
                    'learning_rate': learning_rate,
                    'border_count': border_count,
                    'l2_leaf_reg': l2_leaf_reg,
                    "verbose": 500 # iterations per output
                }

    random_generator = np.random.default_rng()
    overall_model = None
    num_models = 0
    for batch_idx, block_idx in enumerate(train_block_idx):
        print(f"Catboost training batch {batch_idx+1} of {len(train_block_idx)}")
        block_base_row_idx = block_idx * catboost_batch_size
        train_x, train_y, trick_x = dataset.get_np_block_slice(block_base_row_idx, block_base_row_idx + catboost_batch_size)
        train_x = catboost_process_input_batch(train_x, col_data)
        small_random_col = random_generator.random(catboost_batch_size).reshape((catboost_batch_size,1))
        small_random_col *= 1e-20
        train_y=np.where(train_y[:,]==0.0, small_random_col, train_y)
        # Take validation data as last part of this batch; not good because
        # do doubt highly correlated with training data, but to get things working...
        block_model = catboost.CatBoostRegressor(**cat_params)
        block_model.fit(train_x, train_y)
        num_models += 1
        if not overall_model:
            overall_model = block_model
        else:
            new_propn = 1.0 / num_models
            old_propn = 1.0 - new_propn
            overall_model = catboost.sum_models((overall_model, block_model), 
                                                weights=(old_propn, new_propn))
        del block_model
        gc.collect()

    with open(exec_data.model_save_path, "wb") as fd:
        pickle.dump(overall_model, fd)

    # Validation step
    analysis_data = AnalysisData()
    
    for batch_idx, block_idx in enumerate(val_block_idx):
        block_base_row_idx = block_idx * catboost_batch_size
        val_x, val_y, trick_x = dataset.get_np_block_slice(block_base_row_idx, block_base_row_idx + catboost_batch_size)
        val_x = catboost_process_input_batch(val_x, col_data)
        predicted_y = overall_model.predict(val_x)
        r2_score = sklearn.metrics.r2_score(val_y, predicted_y)
        print(f"Catboost validation batch {batch_idx+1} of {len(val_block_idx)} normalised r2={r2_score}")
        analyse_batch(analysis_data, val_x, predicted_y, val_y, trick_x, col_data, scaling_data, submission_weights_old)

        if os.path.exists(stopfile_path):
            print("Stop file detected, deleting it and stopping now")
            os.remove(stopfile_path)
            exec_data.stop_requested = True
            break

    analysis_data.df.head(max_analysis_output_rows).write_csv(analysis_df_path)
    bad_r2_output_names = calc_output_r2_ranking(col_data, analysis_data)
    print(f"Final validation R2={analysis_data.r2_raw}, bad excl R2={analysis_data.r2_clean}")
    with open(loss_log_path, 'a') as fd:
        fd.write(f'{analysis_data.r2_raw},{analysis_data.r2_clean}\n')

    if analysis_data.r2_clean > exec_data.overall_best_val_metric:
        print(f'Best model so far {exec_data.model_save_path}')
        exec_data.overall_best_model_name = exec_data.model_save_path
        exec_data.overall_best_model = overall_model
        exec_data.overall_best_val_metric = analysis_data.r2_clean
        exec_data.best_feature_knockout_idx = exec_data.feature_knockout_idx

    with open(catboost_analysis_data_path, 'wb') as fd:
        pickle.dump(analysis_data, fd)

    return  bad_r2_output_names


def catboost_process_input_batch(x_np, col_data):

    num_rows = x_np.shape[0]
    x_proc = x_np.reshape((num_rows,-1)) # Leaving layer duplicates of scalars

    # Replace blocks of identical values from vectorised scalars with single
    # scalar element they started as
    x_idx = 0
    for catboost_ip_feature_idx in col_data.catboost_input_feature_idx:
        col_name = col_data.unexpanded_input_col_names[catboost_ip_feature_idx]
        col = col_data.unexpanded_cols_by_name[col_name]
        if (col.dimension <= 1):
            x_proc = np.delete(x_proc, range(x_idx + 1, x_idx + num_atm_levels), axis=1)
        x_idx += col.dimension

    return x_proc

# Training loop
class ExecData():
    def __init__(self):
        self.stop_requested = False
        self.overall_best_model = None
        self.overall_best_model_state = None
        self.overall_best_model_name = ""
        self.overall_best_val_metric = 0.0
        self.best_feature_knockout_idx = 0
        self.feature_knockout_idx = 0

def training_loop(train_hf, num_train_rows, col_data, scaling_data, submission_weights_old, param_permutations, device):
    exec_data = ExecData()
    if model_type == "cnn":
        exec_data.overall_best_val_metric = float('inf')
        batch_size = cnn_batch_size
    else:
        exec_data.overall_best_val_metric = float('-inf') # R2 bigger the better
        batch_size = catboost_batch_size

    # Access data in blocks that we can cache efficiently, but on a macro scale access those
    # randomly for training and validation

    # If divides exactly this is OK:
    num_blocks = num_train_rows // batch_size
    assert (num_train_rows / batch_size == num_blocks)
    # Not worrying about using small leftover chunk of rows as yet
    num_full_blocks = num_blocks

    # Not currently using any spillover, because training batch may then start with smaller
    # awkward number of rows and continue into rows belonging into another block, giving
    # a misalgined pattern
    #if num_train_rows % holo_cache_rows != 0:
    #    # Some rows spill over into last less-than-full block
    #    num_blocks += 1
    #    blocks_divide_exactly = False

    block_indices = range(num_blocks)
    train_block_size = int(train_proportion * num_blocks)
    train_block_idx, val_block_idx = sklearn.model_selection.train_test_split(block_indices, train_size=train_block_size)

    train_row_idx = form_row_range_from_block_range(train_block_idx, num_train_rows, num_blocks, num_full_blocks, batch_size)
    val_row_idx = form_row_range_from_block_range(val_block_idx, num_train_rows, num_blocks, num_full_blocks, batch_size)

    cnn_dataset      = HoloDataset(train_hf, cache_batch_size, exec_data, device, col_data.cnn_input_feature_idx)
    catboost_dataset = HoloDataset(train_hf, cache_batch_size, exec_data, device, col_data.catboost_input_feature_idx)

    if model_type == "cnn":
        train_dataset = torch.utils.data.Subset(cnn_dataset, train_row_idx)
        val_dataset = torch.utils.data.Subset(cnn_dataset, val_row_idx)
        train_loader = DataLoader(train_dataset, batch_size=cnn_batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=cnn_batch_size, shuffle=False)

    for param_permutation in param_permutations:
        if os.path.exists(stopfile_path):
            print("Stop file detected, deleting it and stopping now")
            os.remove(stopfile_path)
            exec_data.stop_requested = True
            break

        if exec_data.stop_requested: # can happen inside training loop
            break

        print("Starting training loop...")
        if do_feature_knockout:
            # permutation is just index of feature to knock out
            model_params = {}
            exec_data.feature_knockout_idx = param_permutation
            exec_data.feature_knockout_name = col_data.unexpanded_input_col_names[exec_data.feature_knockout_idx]
            exec_data.feature_knockout_col = col_data.unexpanded_cols_by_name[exec_data.feature_knockout_name]
            exec_data.feature_knockout_description = exec_data.feature_knockout_col.description
            print(f"... knocking out feature {exec_data.feature_knockout_idx}: {exec_data.feature_knockout_name} - {exec_data.feature_knockout_description}")
            suffix = f"_knockout_{exec_data.feature_knockout_idx}_{exec_data.feature_knockout_name}"
        else:
            model_params = param_permutation
            suffix = ""
            for key in param_permutation.keys():
                print(f"... {key}={param_permutation[key]}")
                suffix += f"_{key}_{param_permutation[key]}"
        exec_data.model_save_path = model_root_path + suffix
        if model_type == "cnn":
            exec_data.model_save_path += ".pt"
        else:
            exec_data.model_save_path += ".pkl"

        with open(loss_log_path, 'a') as fd:
            fd.write(f'{exec_data.model_save_path}\n')

        if model_type == "cnn":
            bad_r2_output_names = do_cnn_training(model_params, exec_data, col_data, scaling_data, submission_weights_old,
                                                  param_permutations, train_loader, val_loader, cnn_dataset, device)
        else:
            bad_r2_output_names = do_catboost_training(exec_data, col_data, scaling_data, submission_weights_old, catboost_dataset, train_block_idx,
                                                       val_block_idx, **model_params)

        if do_feature_knockout:
            with open(feature_knockout_path, 'a') as fd:
                fd.write(f"{exec_data.feature_knockout_idx},{exec_data.overall_best_val_metric},{exec_data.feature_knockout_name},{exec_data.feature_knockout_description}\n")

    return exec_data, bad_r2_output_names


def test_submission(col_data, scaling_data, exec_data, bad_r2_output_names, device,
                      submission_weights_current, submission_weights_old):
    print('Loading test HoloFrame...')
    test_hf = HoloFrame(test_path, test_offsets_path)

    submission_df = None

    print(f'Using model {exec_data.overall_best_model_name} for test run and submission')
    if model_type == "cnn":
        exec_data.overall_best_model.load_state_dict(exec_data.overall_best_model_state)
        exec_data.overall_best_model.eval()
 
    print("Removing poor R2 cols from those that will be predicted:", bad_r2_output_names)
    for name in bad_r2_output_names:
        col_data.zero_or_bad_cols_by_name[name] = False

    base_row_idx = 0
    num_test_rows = min(max_test_rows, len(test_hf))
    while base_row_idx < num_test_rows:
        print(f'Processing submission from row {base_row_idx}')
        num_rows = min(len(test_hf) - base_row_idx, test_batch_size)
        subset_df = test_hf.get_slice(base_row_idx, base_row_idx + num_rows)
        base_row_idx += num_rows

        xt_prenorm, _    = preprocess_data(subset_df, False, col_data, scaling_data, submission_weights_current, submission_weights_old)
        xt, _            = normalise_data(xt_prenorm, None, scaling_data, False)
        x_trick_features = extract_raw_trick_subset(xt_prenorm, col_data)

        if do_feature_knockout:
            # Would never really do test run with single feature knocked out, but supporting anyway
            xt = np.delete(xt, exec_data.best_feature_knockout_idx, axis=1)

        if model_type == "cnn":
            # Convert the current slice of xt to a PyTorch tensor
            xt = xt[:, col_data.cnn_input_feature_idx, :]
            inputs = torch.from_numpy(xt).to(device)

            # No need to track gradients for inference
            with torch.no_grad():
                outputs_pred = exec_data.overall_best_model(inputs)
                y_predictions = outputs_pred.cpu().numpy()
        else:
            xt = xt[:, col_data.catboost_input_feature_idx, :]
            xt = catboost_process_input_batch(xt, col_data)
            y_predictions = exec_data.overall_best_model.predict(xt)

        y_predictions = unscale_outputs(y_predictions, scaling_data, submission_weights_old)
        y_predictions = postprocess_predictions(xt, y_predictions, x_trick_features, scaling_data, col_data)

        # We already premultiplied training values by submission weights
        # so predictions should already be scaled the same way

        # Lose everything apart from sample ID:
        submission_subset_df = subset_df.select('sample_id')
        # Add output columns for submission
        submission_subset_df = submission_subset_df.with_columns(pl.from_numpy(y_predictions, col_data.expanded_names_output))

        if submission_df is not None:
            submission_df = pl.concat([submission_df, submission_subset_df])
        else:
            submission_df = submission_subset_df

        gc.collect()

    print("submission_df:", submission_df.describe())

    submission_df.write_csv("submission.csv")

def exit_clean_up():
    if clear_batch_cache_at_end:
        print('Deleting batch cache files...')
        shutil.rmtree(batch_cache_dir)


if __name__ == "__main__":
    # Note: this is true when executed in Kaggle notebook cell too
    main()
