# LEAP competition with feature engineering

# This block will be different in Kaggle notebook:
run_local = True
debug = False
do_test = True

#

if debug:
    max_train_rows = 5000
    max_test_rows  = 1000
    max_batch_size = 1000
    patience = 4
    train_proportion = 0.7
else:
    # Very large numbers for 'all'
    max_train_rows = 10000000 # was using 100000 but saving GPU quota
    max_test_rows  = 1000000000
    max_batch_size = 50000
    patience = 3 # was 5 but saving GPU quota
    train_proportion = 0.75

show_timings = debug
batch_report_interval = 10

holo_cache_rows = max_batch_size # Explore later if helps to cache for multi batches

import copy
import numpy as np
import os
import pickle
import polars as pl
import re
import sklearn.model_selection
import sys
import time

if run_local:
    base_path = '.'
    offsets_path = '.'
    train_root = 'train'
    test_root = 'test'
    submission_root = 'sample_submission-top-10000'
else:
    base_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
    offsets_path = '/kaggle/input/leap-atmospheric-physics-file-row-offsets'
    train_root = 'train'
    test_root = 'test'
    submission_root = 'sample_submission'

train_path = os.path.join(base_path, train_root + '.csv')
train_offsets_path = os.path.join(offsets_path, train_root + '.pkl')
test_path = os.path.join(base_path, test_root + '.csv')
test_offsets_path = os.path.join(offsets_path, test_root + '.pkl')
submission_path = os.path.join(base_path, submission_root + '.csv')
submission_offsets_path = os.path.join(offsets_path, submission_root + '.pkl')

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

# Read in training data
train_sf = HoloFrame(train_path, train_offsets_path)

# First row is all we need from submissions, to get col weightings. 
# sample_id column labels are identical to test.csv (checked first rows at least)
sample_submission_df = pl.read_csv(submission_path, n_rows=1)

# And test data
if do_test:
    test_df = pl.read_csv(test_path, n_rows=max_test_rows)
    print("test_df:", test_df.describe())

# Altitude levels in hPa from ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc
level_pressure_hpa = [0.07834781133863082, 0.1411083184744011, 0.2529232969453412, 0.4492506351686618, 0.7863461614709879, 1.3473557602677517, 2.244777286900205, 3.6164314830257718, 5.615836425337344, 8.403253219853443, 12.144489352066294, 17.016828024303006, 23.21079811610005, 30.914346261995327, 40.277580662953575, 51.37463234765765, 64.18922841394662, 78.63965761131159, 94.63009200213703, 112.09127353988006, 130.97780378937776, 151.22131809551237, 172.67390465199267, 195.08770981962772, 218.15593476138105, 241.60037901222947, 265.2585152868483, 289.12232222921756, 313.31208711045167, 338.0069992368819, 363.37349177951705, 389.5233382784413, 416.5079218282233, 444.3314120123719, 472.9572063769364, 502.2919169181905, 532.1522731583445, 562.2393924639011, 592.1492760575118, 621.4328411158061, 649.689897132655, 676.6564846051039, 702.2421877859194, 726.4985894989197, 749.5376452869328, 771.4452171682528, 792.2342599534793, 811.8566751313328, 830.2596431972574, 847.4506530638328, 863.5359020075301, 878.7158746040692, 893.2460179738746, 907.3852125876941, 921.3543974831824, 935.3167171670306, 949.3780562075774, 963.5995994020714, 978.013432382012, 992.6355435925217]
num_levels = len(level_pressure_hpa)


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

# Data column metadata from https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data
unexpanded_col_list = [
    #         input?   Name                Description                                    Dimension   Units   Useful-from-idx
    ColumnInfo(True,  'state_t',          'air temperature',                                     60, 'K'      ),
    #ColumnInfo(True,  'state_q0001',      'specific humidity',                                   60, 'kg/kg'  ),
    ColumnInfo(True,  'state_q0002',      'cloud liquid mixing ratio',                           60, 'kg/kg'  ),
    ColumnInfo(True,  'state_q0003',      'cloud ice mixing ratio',                              60, 'kg/kg'  ),
    #ColumnInfo(True,  'state_u',          'zonal wind speed',                                    60, 'm/s'    ),
    #ColumnInfo(True,  'state_v',          'meridional wind speed',                               60, 'm/s'    ),
    ColumnInfo(True,  'state_ps',         'surface pressure',                                     1, 'Pa'     ),
    #ColumnInfo(True,  'pbuf_SOLIN',       'solar insolation',                                     1, 'W/m2'   ),
    ColumnInfo(True,  'pbuf_LHFLX',       'surface latent heat flux',                             1, 'W/m2'   ),
    ColumnInfo(True,  'pbuf_SHFLX',       'surface sensible heat flux',                           1, 'W/m2'   ),
    ColumnInfo(True,  'pbuf_TAUX',        'zonal surface stress',                                 1, 'N/m2'   ),
    ColumnInfo(True,  'pbuf_TAUY',        'meridional surface stress',                            1, 'N/m2'   ),
    #ColumnInfo(True,  'pbuf_COSZRS',      'cosine of solar zenith angle',                         1           ),
    #ColumnInfo(True,  'cam_in_ALDIF',     'albedo for diffuse longwave radiation',                1           ),
    #ColumnInfo(True,  'cam_in_ALDIR',     'albedo for direct longwave radiation',                 1           ),
    #ColumnInfo(True,  'cam_in_ASDIF',     'albedo for diffuse shortwave radiation',               1           ),
    #ColumnInfo(True,  'cam_in_ASDIR',     'albedo for direct shortwave radiation',                1           ),
    #ColumnInfo(True,  'cam_in_LWUP',      'upward longwave flux',                                 1, 'W/m2'   ),
    ColumnInfo(True,  'cam_in_ICEFRAC',   'sea-ice areal fraction',                               1           ),
    ColumnInfo(True,  'cam_in_LANDFRAC',  'land areal fraction',                                  1           ),
    ColumnInfo(True,  'cam_in_OCNFRAC',   'ocean areal fraction',                                 1           ),
    ColumnInfo(True,  'cam_in_SNOWHLAND', 'snow depth over land',                                 1, 'm'      ),
    ColumnInfo(True,  'pbuf_ozone',       'ozone volume mixing ratio',                           60, 'mol/mol'),
    ColumnInfo(True,  'pbuf_CH4',         'methane volume mixing ratio',                         60, 'mol/mol'),
    ColumnInfo(True,  'pbuf_N2O',         'nitrous oxide volume mixing ratio',                   60, 'mol/mol'),
    ColumnInfo(False, 'ptend_t',          'heating tendency',                                    60, 'K/s'    ),
    ColumnInfo(False, 'ptend_q0001',      'moistening tendency',                                 60, 'kg/kg/s', 12),
    ColumnInfo(False, 'ptend_q0002',      'cloud liquid mixing ratio change over time',          60, 'kg/kg/s', 15),
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
unexpanded_col_list.append(ColumnInfo(True, 'pressure',          'air pressure',                        60, 'N/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'density',           'air density',                         60, 'kg/m3'      ))
unexpanded_col_list.append(ColumnInfo(True, 'recip_density',     'reciprocal air density',              60, 'm3/kg'      ))
unexpanded_col_list.append(ColumnInfo(True, 'momentum_u',        'zonal momentum per unit volume',      60, '(kg.m/s)/m3'))
unexpanded_col_list.append(ColumnInfo(True, 'momentum_v',        'meridional momentum per unit volume', 60, '(kg.m/s)/m3'))
unexpanded_col_list.append(ColumnInfo(True, 'rel_humidity',      'relative humidity (proportion)'     , 60               ))
unexpanded_col_list.append(ColumnInfo(True, 'vert_insolation',   'zenith-adjusted insolation',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_sw_absorb',  'direct shortwave absorbance',          1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_sw_absorb', 'diffuse shortwave absorbance',         1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_lw_absorb',  'direct longwave absorbance',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_lw_absorb', 'diffuse longwave absorbance',          1, 'W/m2'       ))

unexpanded_col_names = [col.name for col in unexpanded_col_list]
unexpanded_cols_by_name = dict(zip(unexpanded_col_names, unexpanded_col_list))

unexpanded_input_col_names = [col.name for col in unexpanded_col_list if col.is_input]
unexpanded_output_col_names = [col.name for col in unexpanded_col_list if not col.is_input]

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

expanded_col_list = []
expand_and_add_cols(expanded_col_list, unexpanded_cols_by_name, unexpanded_input_col_names)
expand_and_add_cols(expanded_col_list, unexpanded_cols_by_name, unexpanded_output_col_names)

expanded_names = [col.name for col in expanded_col_list]
expanded_names_input = [col.name for col in expanded_col_list if col.is_input]
expanded_names_output = [col.name for col in expanded_col_list if not col.is_input]


# Functions to compute saturation pressure at given temperature taken from
# https://colab.research.google.com/github/tbeucler/CBRAIN-CAM/blob/master/Climate_Invariant_Guide.ipynb#scrollTo=1Hsy9p4Ghe-G
# required to compute relative humidity, which generalises much better than
# specific humidity according to Beucler et al 2021:

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


# Not sure if adding single columns at a time will be too slow on big dataset, can do:
#new_pressure_cols = [pl.lit(level_pressure_hpa[i]).alias(f'pressure_{i}') for i in range(num_levels)]
#train_df = train_df.with_columns(new_pressure_cols)


def add_input_features(df):
    R_air = 287.0 # Mass-based gas constant approx for air in J/kg.K
    for i in range(num_levels):
        # Column names for this level
        cn_pressure       = f'pressure_{i}'      # Pressure in hPa
        cn_temperature    = f'state_t_{i}'       # Temperature in K
        cn_density        = f'density_{i}'       # Density in kg/m3
        cn_recip_density  = f'recip_density_{i}' # Density in kg/m3
        cn_mtm_zonal      = f'momentum_u_{i}'    # Zonal (E-W) momentum per unit volume in kg/m3.m/s
        cn_mtm_meridional = f'momentum_v_{i}'    # Meridional (N-S) momentum per unit volume in kg/m3.m/s
        cn_vel_zonal      = f'state_u_{i}'       # Zonal velocity in m/s
        cn_vel_meridional = f'state_v_{i}'       # Meridional velocity in m/s
        cn_sp_humidity    = f'state_q0001_{i}'   # Specific humidity (kg/kg)
        cn_rel_humidity   = f'rel_humidity_{i}'  # Relative humidity (proportion)

        # Using fixed pressure levels, hopefully near enough, not sure in dataset whether
        # we're supposed to scale with surface pressure or something:
        df = df.with_columns(pl.lit(level_pressure_hpa[i]).alias(cn_pressure))
        # pV = mRT
        # m/V = p/RT = density, with *100 for hPa -> Pa conversion
        df = df.with_columns((pl.col(cn_pressure) * 100.0 / (R_air * pl.col(cn_temperature))).alias(cn_density))
        df = df.with_columns((1.0 / pl.col(cn_density)).alias(cn_recip_density))
        # Momentum per unit vol just density * velocity
        df = df.with_columns((pl.col(cn_density) * pl.col(cn_vel_zonal)).alias(cn_mtm_zonal))
        df = df.with_columns((pl.col(cn_density) * pl.col(cn_vel_meridional)).alias(cn_mtm_meridional))
        # https://www.reddit.com/r/rust/comments/137jcck/polars_computing_a_new_column_from_multiple/
        df = df.with_columns(pl.struct([cn_sp_humidity, cn_temperature, cn_pressure]).map_elements(
                       lambda x: RH_from_climate(x[cn_sp_humidity], x[cn_temperature],
                          x[cn_pressure] * 100.0), return_dtype=pl.datatypes.Float32).alias(cn_rel_humidity))

    # Single-value new features

    # Solar insolation adjusted for zenith angle (angle to vertical)
    df = df.with_columns((pl.col('pbuf_SOLIN') * pl.col('pbuf_COSZRS')).alias('vert_insolation'))
    # Absorbance of solar shortwave radiation
    df = df.with_columns((pl.col('vert_insolation') * (1.0 - pl.col('cam_in_ASDIR'))).alias('direct_sw_absorb'))
    df = df.with_columns((pl.col('vert_insolation') * (1.0 - pl.col('cam_in_ASDIF'))).alias('diffuse_sw_absorb'))
    # Absorbance of IR radiation from ground
    df = df.with_columns((pl.col('cam_in_LWUP') * (1.0 - pl.col('cam_in_ALDIR'))).alias('direct_lw_absorb'))
    df = df.with_columns((pl.col('cam_in_LWUP') * (1.0 - pl.col('cam_in_ALDIF'))).alias('diffuse_lw_absorb'))

    return df


re_vector_heading = re.compile('([A-Za-z0-9_]+?)_([0-9]+)')

def vectorise_data(pl_df):
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
            root_str = matcher.group(1)
            idx_str = matcher.group(2)
            idx = int(idx_str)
            if idx != 0:
                print(f"Error: expected zeroth element first, got {idx}")
                sys.exit(1)
            df_level_subset = pl_df[: , col_idx : col_idx + num_atm_levels]
            row_level_matrix = df_level_subset.to_numpy().astype(np.float64)
            col_idx += num_atm_levels
        else:
            row_vector = pl_df[: , col_name].to_numpy().astype(np.float64)
            row_vector = row_vector.reshape(len(row_vector), 1)
            row_level_matrix = np.tile(row_vector, (1,num_atm_levels))
            col_idx += 1
        vector_dict[root_str] = row_level_matrix

    return vector_dict


mx_sample = [] # Each element vector of means of input columns, from one holo batch
sx_sample = [] # ... and scaling factor
my_sample = [] # Similarly for output columns
sy_sample = []

def mean_vector_across_samples(sample_list):
    """Given series of sample row vectors across data columns, form
    new row vector which is mean of those samples"""

    # join series of row vectors across inputs to one matrix of row samples vs column inputs
    all_samples = np.stack(sample_list)
    # Get mean down each column to get new row vector across input columns
    mean_vector = all_samples.mean(axis=0)
    return mean_vector

def preprocess_data(pl_df, has_outputs):
    if debug:
        vector_dict = vectorise_data(pl_df)
    pl_df = add_input_features(pl_df)
    x = pl_df[expanded_names_input].to_numpy().astype(np.float64)
    if has_outputs:
        y = pl_df[expanded_names_output].to_numpy().astype(np.float64)
    del pl_df

    # norm X
    if has_outputs:
        mx = x.mean(axis=0)
        sx = np.maximum(x.std(axis=0), min_std)
        mx_sample.append(mx)
        sx_sample.append(sx)
    else:
        mx = mean_vector_across_samples(mx_sample)
        sx = mean_vector_across_samples(sx_sample)

    # Original had mx.reshape(1,-1) to go from 1D row vector to 2D array with
    # one row but seems unnecessary
    x = (x - mx) / sx
    x = x.astype(np.float32)

    if has_outputs:
        # norm Y
        # Scaling outputs by weights that wil be used anyway for submission, so we get
        # rid of very tiny values that will give very small variances, and will make
        # them suitable hopefully for conversion to float32
        y = y * submission_weights

        my = y.mean(axis=0)
        # Donor notebook used RMS instead of stdev here, discussion thread suggesting that
        # gives loss value like competition criterion but I see no training advantage:
        # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/498806
        sy = np.maximum(y.std(axis=0), min_std)
        my_sample.append(my)
        sy_sample.append(sy)
        y = y - my / sy
        y = y.astype(np.float32)
    else:
        y = None

    return x, y


# Now trying same as public notebook but with the new features:
# https://www.kaggle.com/code/airazusta014/pytorch-nn/notebook

import random, gc, warnings
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(42)
random.seed(42)
min_std = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUGGING = not do_test

#


# Single row of weights for outputs
submission_weights = sample_submission_df[expanded_names_output].to_numpy().astype(np.float64)
del sample_submission_df

gc.collect()

#


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        
        # Initialize the layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            # nn.Linear: classic weights & biases neuron
            layers.append(nn.Linear(previous_size, hidden_size))
            # LayerNorm: outputs rescaled to mean zero, unit variance across vector
            # of all features at this iteration (as opposed to batch normalisation when
            # features normalised separately in parallel according to their values
            # across a series of samples in the batch)
            # https://www.pinecone.io/learn/batch-layer-normalization/
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer
            # LeakyReLU: non-zero shallow scaling for -ve input values cf classic
            # ReLU which is zero for -ve, default 0.01 gradient on -ve side
            #layers.append(nn.LeakyReLU(inplace=True))   # Activation
            layers.append(nn.SiLU(inplace=True))        # Activation
            layers.append(nn.Dropout(p=0.1))            # Dropout for regularization
            previous_size = hidden_size
        
        # Output layer - no dropout, no activation function
        layers.append(nn.Linear(previous_size, output_size))
        
        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
#

class HoloDataset(Dataset):
    def __init__(self, holo_df, cache_rows):
        """
        Initialize with HoloFrame instance.
        """
        self.holo_df = holo_df
        self.cache_rows = cache_rows
        self.cache_base_idx = -1
        self.cache_np_x = None
        self.cache_np_y = None

    def __len__(self):
        """
        Total number of samples.
        """
        return len(self.holo_df)

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        if not (self.cache_base_idx >= 0 and 
                index >= self.cache_base_idx and index < self.cache_base_idx + self.cache_rows):
            # We don't already have this convered and processed, so process
            # a batch of rows to cache
            self.cache_base_idx = index
            start_time = time.time()
            pl_slice_df = self.holo_df.get_slice(index, index + self.cache_rows)
            self.cache_np_x, self.cache_np_y = preprocess_data(pl_slice_df, True)
            if show_timings: print(f'HoloDataset slice build at row {self.cache_base_idx} took {time.time() - start_time} s')

        # Convert the data to tensors when requested
        cache_idx = index - self.cache_base_idx
        return torch.from_numpy(self.cache_np_x[cache_idx]).float().to(device), \
               torch.from_numpy(self.cache_np_y[cache_idx]).float().to(device)
    
#

dataset = HoloDataset(train_sf, holo_cache_rows)
num_train_rows = min(len(dataset), max_train_rows)

# Access data in blocks that we can cache efficiently, but on a macro scale access those
# randomly for training and validation

# If divides exactly this is OK:
num_blocks = num_train_rows // holo_cache_rows
blocks_divide_exactly = True
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

def form_row_range_from_block_range(block_indices):
    row_idx = []
    for block_idx in block_indices:
        if block_idx == num_blocks - 1 and not blocks_divide_exactly:
            # Smaller number in last dangling block
            num_rows_in_block = num_train_rows - num_full_blocks * holo_cache_rows
        else:
            num_rows_in_block = holo_cache_rows
        base_row_idx = block_idx * holo_cache_rows
        row_idx.extend(list(range(base_row_idx, base_row_idx + num_rows_in_block)))
    return row_idx

train_row_idx = form_row_range_from_block_range(train_block_idx)
val_row_idx = form_row_range_from_block_range(val_block_idx)

train_dataset = torch.utils.data.Subset(dataset, train_row_idx)
val_dataset = torch.utils.data.Subset(dataset, val_row_idx)
train_loader = DataLoader(train_dataset, batch_size=max_batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=max_batch_size, shuffle=False)

input_size = len(expanded_names_input) # number of input features/columns
output_size = len(expanded_names_output)
hidden_size = input_size + output_size # any particular reason for this and 'diabalo' shape here?
model = FFNN(input_size, [3*hidden_size, 2*hidden_size, hidden_size, 2*hidden_size, 3*hidden_size], output_size).to(device)
criterion = nn.MSELoss()  # Using MSE for regression
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)


#

# Training loop
epochs = 100000
best_val_loss = float('inf')  # Set initial best as infinity
best_model_state = None       # To store the best model's state
patience_count = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    steps = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Calculates gradients by backpropagation (chain rule)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if show_timings: print(f'Training batch of {max_batch_size} took {time.time() - start_time} s')

        # Print every n steps
        if (batch_idx + 1) % batch_report_interval == 0:
            print(f'Epoch {epoch + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}')
            total_loss = 0  # Reset the loss for the next n steps
            steps = 0  # Reset step count
    

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')
    
    scheduler.step(avg_val_loss)  # Adjust learning rate

    # Update best model if current epoch's validation loss is lower
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()  # Save the best model state
        patience_count = 0
        print("Validation loss decreased, saving new best model and resetting patience counter.")
    else:
        patience_count += 1
        print(f"No improvement in validation loss for {patience_count} epochs.")
        
    if patience_count >= patience:
        print("Stopping early due to no improvement in validation loss.")
        break


#

# Test
if not DEBUGGING:
    xt, _ = preprocess_data(test_df, False)

    model.load_state_dict(best_model_state)
    model.eval()
    predt = np.zeros([xt.shape[0], output_size], dtype=np.float32)  # output_size is the dimension of your model's output
    batch_size = 1024 * 128  # Batch size for inference

    i1 = 0
    for i in range(10000):
        i2 = np.minimum(i1 + batch_size, xt.shape[0])
        if i1 == i2:  # Break the loop if range does not change
            break

        # Convert the current slice of xt to a PyTorch tensor
        inputs = torch.from_numpy(xt[i1:i2, :]).float().to(device)

        # No need to track gradients for inference
        with torch.no_grad():
            outputs = model(inputs)  # Get model predictions
            predt[i1:i2, :] = outputs.cpu().numpy()  # Store predictions in predt

        print(np.round(i2 / predt.shape[0], 2))  # Print the percentage completion
        i1 = i2  # Update i1 to the end of the current batch

        if i2 >= xt.shape[0]:
            break


#

if not DEBUGGING:
    # submit
    # override constant columns
    my = mean_vector_across_samples(my_sample)
    sy = mean_vector_across_samples(sy_sample)

    for i in range(sy.shape[0]):
        # CW: still using original threshold although now premultiplying outputs by
        # submission weightings, though does zero out those with zero weights
        # (and some others)
        if sy[i] < min_std * 1.1:
            predt[:,i] = 0 # Although zero here after rescaling will be y.mean
            print("Using mean for:", expanded_names_output[i])

    # undo y scaling
    predt = predt * sy + my

    # We already premultiplied training values by submission weights
    # so predictions should already be scaled the same way

    submission_df = pl.DataFrame(test_df['sample_id'])
    submission_df = submission_df.with_columns(pl.from_numpy(predt, expanded_names_output))

    print("submission_df:", submission_df.describe())

    if debug:
        submission_df.write_csv("submission-debug.csv")
    else:
        submission_df.write_csv("submission.csv")

pass
