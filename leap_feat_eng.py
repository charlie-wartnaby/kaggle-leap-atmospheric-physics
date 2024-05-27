# LEAP competition with feature engineering

# This block will be different in Kaggle notebook:
run_local = True
debug = False
do_test = True
use_cnn = True
is_rerun = True
is_analysis = True

#

if debug:
    max_train_rows = 5000
    max_test_rows  = 1000
    max_batch_size = 1000
    patience = 4
    train_proportion = 0.8
    max_epochs = 10
else:
    # Use very large numbers for 'all'
    max_train_rows = 100
    max_test_rows  = 1000000000
    max_batch_size = 10  # 5000 with pcuk151, 30000 greta
    patience = 3 # was 5 but saving GPU quota
    train_proportion = 0.9
    max_epochs = 50

multitrain_params = {}

show_timings = False # debug
batch_report_interval = 10
dropout_p = 0.1
initial_learning_rate = 0.001 # default 0.001
try_reload_model = is_rerun or is_analysis
clear_batch_cache_at_start = False # forgot before starting # not is_rerun #debug # True if processing has changed
clear_batch_cache_at_end = False # not debug -- save Kaggle quota by deleting there?
if is_analysis: max_epochs = 1
max_analysis_output_rows = 1000
holo_cache_rows = max_batch_size # Explore later if helps to cache for multi batches

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

import copy
import numpy as np
import os
import pickle
import polars as pl
import re
import shutil
import sklearn.model_selection
import sys
import time

import socket
machine = socket.gethostname()

if machine == 'narg':
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

train_path = os.path.join(base_path, train_root + '.csv')
train_offsets_path = os.path.join(offsets_path, train_root + '.pkl')
test_path = os.path.join(base_path, test_root + '.csv')
test_offsets_path = os.path.join(offsets_path, test_root + '.pkl')
submission_template_path = os.path.join(base_path, submission_root + '.csv')
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

stopfile_path = 'stop.txt'

if os.path.exists(stopfile_path):
    print("Stop file detected on entry, deleting it")
    os.remove(stopfile_path)
if not is_rerun and os.path.exists(epoch_counter_path):
    os.remove(epoch_counter_path)
if not is_rerun and os.path.exists(loss_log_path):
    os.remove(loss_log_path)

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
print('Loading training HoloFrame...')
train_hf = HoloFrame(train_path, train_offsets_path)

# First row is all we need from submissions, to get col weightings. 
# sample_id column labels are identical to test.csv (checked first rows at least)
print('Loading submission weights...')
sample_submission_df = pl.read_csv(submission_template_path, n_rows=1)

if clear_batch_cache_at_start and os.path.exists(batch_cache_dir):
    print('Deleting any previous batch cache files...')
    shutil.rmtree(batch_cache_dir)
os.makedirs(batch_cache_dir, exist_ok=True)

# Altitude levels in hPa from ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc
level_pressure_hpa = [0.07834781133863082, 0.1411083184744011, 0.2529232969453412, 0.4492506351686618, 0.7863461614709879, 1.3473557602677517, 2.244777286900205, 3.6164314830257718, 5.615836425337344, 8.403253219853443, 12.144489352066294, 17.016828024303006, 23.21079811610005, 30.914346261995327, 40.277580662953575, 51.37463234765765, 64.18922841394662, 78.63965761131159, 94.63009200213703, 112.09127353988006, 130.97780378937776, 151.22131809551237, 172.67390465199267, 195.08770981962772, 218.15593476138105, 241.60037901222947, 265.2585152868483, 289.12232222921756, 313.31208711045167, 338.0069992368819, 363.37349177951705, 389.5233382784413, 416.5079218282233, 444.3314120123719, 472.9572063769364, 502.2919169181905, 532.1522731583445, 562.2393924639011, 592.1492760575118, 621.4328411158061, 649.689897132655, 676.6564846051039, 702.2421877859194, 726.4985894989197, 749.5376452869328, 771.4452171682528, 792.2342599534793, 811.8566751313328, 830.2596431972574, 847.4506530638328, 863.5359020075301, 878.7158746040692, 893.2460179738746, 907.3852125876941, 921.3543974831824, 935.3167171670306, 949.3780562075774, 963.5995994020714, 978.013432382012, 992.6355435925217]
num_levels = len(level_pressure_hpa)
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
unexpanded_col_list.append(ColumnInfo(True, 'buoyancy',          'Beucler buoyancy metric',             60               ))
unexpanded_col_list.append(ColumnInfo(True, 'vert_insolation',   'zenith-adjusted insolation',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_sw_absorb',  'direct shortwave absorbance',          1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_sw_absorb', 'diffuse shortwave absorbance',         1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_lw_absorb',  'direct longwave absorbance',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_lw_absorb', 'diffuse longwave absorbance',          1, 'W/m2'       ))

unexpanded_col_names = [col.name for col in unexpanded_col_list]
unexpanded_cols_by_name = dict(zip(unexpanded_col_names, unexpanded_col_list))
unexpanded_input_col_names = [col.name for col in unexpanded_col_list if col.is_input]
unexpanded_output_col_names = [col.name for col in unexpanded_col_list if not col.is_input]
unexpanded_output_vector_col_names = [col.name for col in unexpanded_col_list if not col.is_input and col.dimension > 1]
unexpanded_output_scalar_col_names = [col.name for col in unexpanded_col_list if not col.is_input and col.dimension <= 1]

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
expanded_cols_by_name = dict(zip(expanded_names, expanded_col_list))
expanded_names_input = [col.name for col in expanded_col_list if col.is_input]
expanded_names_output = [col.name for col in expanded_col_list if not col.is_input]

num_all_outputs_as_vectors = len(unexpanded_output_col_names)
num_pure_vector_outputs = len(unexpanded_output_vector_col_names)
num_scalar_outputs = len(unexpanded_output_scalar_col_names)
num_total_expanded_outputs = len(expanded_names_output)

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

def add_vector_features(vector_dict):
    R_air = 287.0 # Mass-based gas constant approx for air in J/kg.K
    # cn_pressure       = f'pressure_{i}'      # Pressure in hPa
    # cn_temperature    = f'state_t_{i}'       # Temperature in K
    # cn_density        = f'density_{i}'       # Density in kg/m3
    # cn_recip_density  = f'recip_density_{i}' # Density in kg/m3
    # cn_mtm_zonal      = f'momentum_u_{i}'    # Zonal (E-W) momentum per unit volume in kg/m3.m/s
    # cn_mtm_meridional = f'momentum_v_{i}'    # Meridional (N-S) momentum per unit volume in kg/m3.m/s
    # cn_vel_zonal      = f'state_u_{i}'       # Zonal velocity in m/s
    # cn_vel_meridional = f'state_v_{i}'       # Meridional velocity in m/s
    # cn_sp_humidity    = f'state_q0001_{i}'   # Specific humidity (kg/kg)
    # cn_rel_humidity   = f'rel_humidity_{i}'  # Relative humidity (proportion)

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
    recip_density_np = 1.0 / density_np
    vector_dict['recip_density'] = density_np
    # Momentum per unit vol just density * velocity
    vel_zonal_np = vector_dict['state_u']
    momentum_u_np = density_np * vel_zonal_np
    vector_dict['momentum_u'] = momentum_u_np
    vel_meridional_np = vector_dict['state_v']
    momentum_v_np = density_np * vel_meridional_np
    vector_dict['momentum_v'] = momentum_v_np
    specific_humidity_np = vector_dict['state_q0001']
    rel_humidity_np = RH_from_climate(specific_humidity_np, temperature_np, pressure_np)
    vector_dict['rel_humidity'] = rel_humidity_np
    buoyancy_np = bmse_calc(temperature_np, specific_humidity_np, pressure_np)
    vector_dict['buoyancy'] = buoyancy_np

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
            row_vector = pl_df[: , col_name].to_numpy().astype(np.float64)
            row_vector = row_vector.reshape(len(row_vector), 1)
            col_info = unexpanded_cols_by_name.get(col_name)
            if col_info and not col_info.is_input and col_info.dimension <= 1:
                # Leave output scalars as they are
                row_level_matrix = row_vector
            else:
               row_level_matrix = np.tile(row_vector, (1,num_atm_levels))
            col_idx += 1
        vector_dict[col_name] = row_level_matrix

    return vector_dict

# Cache normalisation data needed for any rerun later
scaling_cache_filename = 'scaling_normalisation.pkl'
scaling_cache_path = os.path.join(batch_cache_dir, scaling_cache_filename)
if os.path.exists(scaling_cache_path):
    print("Opening previous scalings...")
    with open(scaling_cache_path, 'rb') as fd:
        (mx_sample, sx_sample, sy_sample) = pickle.load(fd)
else:
    print("No previous scalings so starting afresh")
    mx_sample = [] # Each element vector of means of input columns, from one holo batch
    sx_sample = [] # ... and scaling factor
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
    if use_cnn:
        vector_dict = vectorise_data(pl_df)
        add_vector_features(vector_dict)
        # Glue input columns together by row, then feature, then atm level
        # (Hope this will work with torch.nn.Conv1d taking N, C, L)
        for i, col_name in enumerate(unexpanded_input_col_names):
            if i == 0:
                x = vector_dict[col_name]
                (rows,cols) = x.shape
                x = x.reshape(rows, 1, cols)
            else:
                x = np.concatenate((x, vector_dict[col_name].reshape(rows, 1, cols)), axis=1)
    else:
        pl_df = add_input_features(pl_df)
        x = pl_df[expanded_names_input].to_numpy().astype(np.float64)

    if has_outputs:
        if use_cnn:
            # Leaving y-data expanded, will need to design model to explode CNN output across
            # columns, as also need to cope with scalar outputs
            for i, col_name in enumerate(unexpanded_output_col_names):
                if i == 0:
                    y = vector_dict[col_name]
                else:
                    y_add = vector_dict[col_name]
                    y = np.concatenate((y, y_add), axis=1)
        else:
            y = pl_df[expanded_names_output].to_numpy().astype(np.float64)

    del pl_df

    # norm X
    if has_outputs:
        # Now applying same scaling across whole 60-level channel, for x at least:
        mx = x.mean(axis=(0,2))
        mx = mx.reshape(1, len(unexpanded_input_col_names), 1)
        sx = np.maximum(x.std(axis=(0,2)), min_std)
        sx = sx.reshape(1, len(unexpanded_input_col_names), 1)
        mx_sample.append(mx)
        sx_sample.append(sx)
    else:
        # Using scaling found in training data; though could use test data if big enough?
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

        # Donor notebook used RMS instead of stdev here, discussion thread suggesting that
        # gives loss value like competition criterion but I see no training advantage:
        # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/498806
        sy = np.maximum(y.std(axis=0), min_std)
        sy_sample.append(sy)
        y /= sy
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

class AtmLayerCNN(nn.Module):
    def __init__(self, gen_conv_width=7, gen_conv_depth=15, init_1x1=False, 
                 norm_type="layer", activation_type="silu"):
        super().__init__()
        
        self.init_1x1 = init_1x1
        self.norm_type = norm_type
        self.activation_type = activation_type

        num_input_feature_chans = len(unexpanded_input_col_names)

        # Initialize the layers

        input_size = num_input_feature_chans

        if self.init_1x1:
            # Having single-width layer just means mixing combinations of input
            # vars for this layer only as a useful starting point
            # Could generalise to multiple if works
            output_size = num_input_feature_chans * gen_conv_depth
            self.conv_layer_1x1 = nn.Conv1d(input_size, output_size, 1,
                                    padding='same')
            self.layernorm_1x1 = self.norm_layer(output_size, num_atm_levels)
            self.activation_layer_1x1 = self.activation_layer()
            self.dropout_layer_1x1 = nn.Dropout(p=dropout_p)
        else:
            # No initial unit width layer
            output_size = input_size

        input_size = output_size
        output_size = num_input_feature_chans * gen_conv_depth
        self.conv_layer_0 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same')
        self.layernorm_0 = self.norm_layer(output_size, num_atm_levels)
        self.activation_layer_0 = self.activation_layer()
        self.dropout_layer_0 = nn.Dropout(p=dropout_p)

        input_size = output_size
        output_size = num_input_feature_chans * gen_conv_depth
        self.conv_layer_1 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same')
        self.layernorm_1 = self.norm_layer(output_size, num_atm_levels)
        self.activation_layer_1 = self.activation_layer()
        self.dropout_layer_1 = nn.Dropout(p=dropout_p)

        input_size = output_size
        self.last_conv_depth = gen_conv_depth
        output_size = num_all_outputs_as_vectors * self.last_conv_depth
        self.conv_layer_2 = nn.Conv1d(input_size, output_size, gen_conv_width,
                                padding='same')
        self.layernorm_2 = self.norm_layer(output_size, num_atm_levels)
        self.activation_layer_2 = self.activation_layer()
        self.dropout_layer_2 = nn.Dropout(p=dropout_p)

        # Output layer - no dropout, no activation function
        # Data is structured such that all vector columns come first, then scalars
        input_size = output_size
        self.conv_vector_harvest = nn.Conv1d(num_pure_vector_outputs * gen_conv_depth,
                                         num_pure_vector_outputs, 1, padding='same')
        self.vector_flatten = nn.Flatten()
        self.pre_scalar_pool = nn.AvgPool1d(num_atm_levels)
        self.scalar_flatten = nn.Flatten()
        self.linear_scalar_harvest = nn.Linear(num_scalar_outputs * gen_conv_depth, num_scalar_outputs)


    def norm_layer(self, num_features, num_chans):
            match self.norm_type:
                case "layer":
                    return nn.LayerNorm([num_features, num_chans])
                case "batch":
                    return nn.BatchNorm1d(num_features)
                case _:
                    print(f"Unexpected norm_type={self.norm_type}")
                    sys.exit(1)

    def activation_layer(self):
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
        num_conv_outputs_for_vectors = num_pure_vector_outputs * self.last_conv_depth
        vector_subset = x[:, : num_conv_outputs_for_vectors, :]
        scalar_subset = x[:, num_conv_outputs_for_vectors :, :]
        scalars_pooled = self.pre_scalar_pool(scalar_subset)
        scalars_flattened = self.scalar_flatten(scalars_pooled)
        scalar_harvest = self.linear_scalar_harvest(scalars_flattened)
        vector_harvest = self.conv_vector_harvest(vector_subset)
        vectors_flattened = self.vector_flatten(vector_harvest)
        expanded_outputs = torch.cat((vectors_flattened, scalar_harvest), dim=1)
        return expanded_outputs
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
            # We don't already have this converted and processed in RAM
            start_time = time.time()
            self.cache_base_idx = index
            cache_filename = f'{self.cache_base_idx}.pkl'
            cache_path = os.path.join(batch_cache_dir, cache_filename)
            if os.path.exists(cache_path):
                # Preprocessing already done, retrieve from binary cache file
                with open(cache_path, 'rb') as fd:
                    (self.cache_np_x, self.cache_np_y) = pickle.load(fd)
                if show_timings: print(f'HoloDataset slice cache load at row {self.cache_base_idx} took {time.time() - start_time} s')    
            else:
                # Process slice of large dataframe corresponding to batch
                pl_slice_df = self.holo_df.get_slice(index, index + self.cache_rows)
                self.cache_np_x, self.cache_np_y = preprocess_data(pl_slice_df, True)
                if show_timings: print(f'HoloDataset slice build at row {self.cache_base_idx} took {time.time() - start_time} s')
                start_time = time.time()
                with open(cache_path, 'wb') as fd:
                    pickle.dump((self.cache_np_x, self.cache_np_y), fd)
                # Cache normalisation data needed for any rerun later
                with open(scaling_cache_path, 'wb') as fd:
                    pickle.dump((mx_sample, sx_sample, sy_sample), fd)

                if show_timings: print(f'HoloDataset slice cache save at row {self.cache_base_idx} took {time.time() - start_time} s')

        # Convert the RAM numpy data to tensors when requested
        cache_idx = index - self.cache_base_idx
        return torch.from_numpy(self.cache_np_x[cache_idx]).float().to(device), \
               torch.from_numpy(self.cache_np_y[cache_idx]).float().to(device)
    
#

dataset = HoloDataset(train_hf, holo_cache_rows)
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


def unscale_outputs(y, sy):
    """Undo normalisation to return to true values (but with submission
    weights still multiplied in)"""

    for i in range(sy.shape[0]):
        # CW: still using original threshold although now premultiplying outputs by
        # submission weightings, though does zero out those with zero weights
        # (and some others)
        if sy[i] < min_std * 1.1:
            y[:,i] = 0 # Although zero here after rescaling will be y.mean

    # undo y scaling
    y *= sy



def create_analysis_dataframe():
    """Output file to show which columns contributing to loss"""
    schema_col_names = expanded_names_input
    for expanded_output in expanded_names_output:
        schema_col_names.append(expanded_output + '_act')
        schema_col_names.append(expanded_output + '_pred')
        schema_col_names.append(expanded_output + '_r2')
    for vector_output in unexpanded_output_vector_col_names:
        schema_col_names.append(vector_output + '_r2avg')

    analysis_df = pl.DataFrame(schema=schema_col_names)
    return analysis_df

def analyse_batch(analysis_df, inputs, outputs_pred, outputs_true):
    """Analyse batch of true versus predicted outputs"""

    # Return to original output scalings (but with submission weights
    # multiplied in) to match competition metric
    sy = mean_vector_across_samples(sy_sample) # 
    outputs_pred_np = outputs_pred.cpu().numpy()
    outputs_true_np = outputs_true.cpu().numpy()
    unscale_outputs(outputs_pred_np, sy)
    unscale_outputs(outputs_true_np, sy)

    # Assuming variance of dataset outputs foudn in training more 
    # representative than variance in this small batch?
    true_variance_sqd = sy * sy # undo sqrt in stdev

    error_residues = outputs_true_np - outputs_pred_np
    error_variance_sqd = np.square(error_residues)
    avg_error_variance_sqd = np.mean(error_variance_sqd, axis=0)
    r2_metric = 1.0 - (avg_error_variance_sqd / true_variance_sqd)
    
    pass


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

# Training loop
stop_requested = False
overall_best_model = None
overall_best_model_state = None
overall_best_model_name = ""
overall_best_val_loss = float('inf')

for param_permutation in param_permutations:
    if stop_requested:
        break

    if is_analysis:
        analysis_df = create_analysis_dataframe()

    print("Starting training loop...")
    suffix = ""
    for key in param_permutation.keys():
        print(f"... {key}={param_permutation[key]}")
        suffix += f"_{key}_{param_permutation[key]}"
    model_save_path = model_root_path + suffix + ".pt"
    with open(loss_log_path, 'a') as fd:
        fd.write(f'{model_save_path}\n')

    if use_cnn:
        model = AtmLayerCNN(**param_permutation).to(device)
    else:
        hidden_size = input_size + output_size # any particular reason for this and 'diabalo' shape here?
        model = FFNN(input_size, [3*hidden_size, 2*hidden_size, hidden_size, 2*hidden_size, 3*hidden_size], output_size).to(device)

    if try_reload_model and os.path.exists(model_save_path):
        print('Attempting to reload model from disk...')
        model.load_state_dict(torch.load(model_save_path))

    criterion = nn.MSELoss()  # Using MSE for regression
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')  # Set initial best as infinity
    best_model_state = None       # To store the best model's state
    patience_count = 0

    if len(param_permutations) > 1:
        tot_epochs = 0

    for epoch in range(max_epochs):
        if not is_analysis:
            model.train()
            total_loss = 0
            steps = 0
            for batch_idx, (inputs, outputs_true) in enumerate(train_loader):
                start_time = time.time()
                optimizer.zero_grad()
                outputs_pred = model(inputs)
                loss = criterion(outputs_pred, outputs_true)
                loss.backward() # Calculates gradients by backpropagation (chain rule)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

                if show_timings: print(f'Training batch of {max_batch_size} took {time.time() - start_time} s')

                # Print every n steps
                if (batch_idx + 1) % batch_report_interval == 0:
                    print(f'Epoch {tot_epochs + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}')
                    total_loss = 0  # Reset the loss for the next n steps
                    steps = 0  # Reset step count
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, outputs_true) in enumerate(val_loader):
                outputs_pred = model(inputs)
                val_loss += criterion(outputs_pred, outputs_true).item()
                if is_analysis:
                    analyse_batch(analysis_df, inputs, outputs_pred, outputs_true)

                if (batch_idx + 1) % batch_report_interval == 0:
                    print(f'Validation batch {batch_idx + 1}')

        avg_val_loss = val_loss / len(val_loader)

        tot_epochs += 1
        with open(epoch_counter_path, 'w') as fd:
            fd.write(f'{tot_epochs}\n')

        print(f'Epoch {tot_epochs}, Validation Loss: {avg_val_loss}')
        with open(loss_log_path, 'a') as fd:
            fd.write(f'{tot_epochs},{avg_val_loss}\n')

        
        scheduler.step(avg_val_loss)  # Adjust learning rate

        if not is_analysis:
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                overall_best_model = model
                overall_best_model_state = model.state_dict() # TODO is this static anyway?
                overall_best_model_name = model_save_path
                print(f"{model_save_path} best so far")

            # Update best model if current epoch's validation loss is lower
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()  # Save the best model state
                patience_count = 0
                print("Validation loss decreased, saving new best model and resetting patience counter.")
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_count += 1
                print(f"No improvement in validation loss for {patience_count} epochs.")
                
            if patience_count >= patience:
                print("Stopping early due to no improvement in validation loss.")
                break

        if os.path.exists(stopfile_path):
            print("Stop file detected, deleting it and stopping now")
            os.remove(stopfile_path)
            stop_requested = True
            break


# Test
if do_test:
    print('Loading test HoloFrame...')
    test_hf = HoloFrame(test_path, test_offsets_path)

    submission_df = None

    sy = mean_vector_across_samples(sy_sample)

    print(f'Using model {overall_best_model_name} for test run and submission')
    overall_best_model.load_state_dict(overall_best_model_state)
    overall_best_model.eval()
 
    base_row_idx = 0
    num_test_rows = min(max_test_rows, len(test_hf))
    while base_row_idx < num_test_rows:
        print(f'Processing submission from row {base_row_idx}')
        num_rows = min(len(test_hf) - base_row_idx, max_batch_size)
        subset_df = test_hf.get_slice(base_row_idx, base_row_idx + num_rows)
        base_row_idx += num_rows

        xt, _ = preprocess_data(subset_df, False)

        # Convert the current slice of xt to a PyTorch tensor
        inputs = torch.from_numpy(xt).float().to(device)

        # No need to track gradients for inference
        with torch.no_grad():
            outputs_pred = overall_best_model(inputs)
            y_predictions = outputs_pred.cpu().numpy()

        unscale_outputs(y_predictions, sy)

        # We already premultiplied training values by submission weights
        # so predictions should already be scaled the same way

        # Lose everything apart from sample ID:
        submission_subset_df = subset_df.select('sample_id')
        # Add output columns for submission
        submission_subset_df = submission_subset_df.with_columns(pl.from_numpy(y_predictions, expanded_names_output))

        if submission_df is not None:
            submission_df = pl.concat([submission_df, submission_subset_df])
        else:
            submission_df = submission_subset_df

    print("submission_df:", submission_df.describe())

    submission_df.write_csv("submission.csv")


if clear_batch_cache_at_end:
    print('Deleting batch cache files...')
    shutil.rmtree(batch_cache_dir)
