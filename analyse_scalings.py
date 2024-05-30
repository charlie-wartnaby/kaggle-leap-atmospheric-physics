import copy
import numpy as np
import os
import pickle
import polars as pl

batch_cache_dir = 'batch_cache'
scaling_cache_filename = 'scaling_normalisation.pkl'
scaling_cache_path = os.path.join(batch_cache_dir, scaling_cache_filename)
if os.path.exists(scaling_cache_path):
    print("Opening previous scalings...")
    with open(scaling_cache_path, 'rb') as fd:
        (mx_sample, sx_sample, sy_sample) = pickle.load(fd)

max_train_rows = 3000000
max_batch_size = 5000
num_batches = int(max_train_rows / max_batch_size)
assert(len(mx_sample) == num_batches)
mx_sample_np = np.array(mx_sample) # now array shape (num batches, 1, unexpanded features, 1)
sx_sample_np = np.array(sx_sample) # ..
sy_sample_np = np.array(sy_sample) ## array (num batches, expanded features)

# Duplicating current col info processing to get names

do_feature_knockout = True # to get complete col list

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

if do_feature_knockout:
    current_normal_knockout_features = []
else:
    current_normal_knockout_features = ['state_q0001', 'state_u', 'state_v', 'pbuf_SOLIN', 'pbuf_COSZRS',
                                    'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP']

for feature in current_normal_knockout_features:
    # Slow but trivial one-off
    for i in range(len(unexpanded_col_list)):
        if unexpanded_col_list[i].name == feature:
            del unexpanded_col_list[i]
            break

# Add columns for new features
unexpanded_col_list.append(ColumnInfo(True, 'pressure',             'air pressure',                        60, 'N/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'density',              'air density',                         60, 'kg/m3'      ))
unexpanded_col_list.append(ColumnInfo(True, 'recip_density',        'reciprocal air density',              60, 'm3/kg'      ))
unexpanded_col_list.append(ColumnInfo(True, 'momentum_u',           'zonal momentum per unit volume',      60, '(kg.m/s)/m3'))
unexpanded_col_list.append(ColumnInfo(True, 'momentum_v',           'meridional momentum per unit volume', 60, '(kg.m/s)/m3'))
unexpanded_col_list.append(ColumnInfo(True, 'rel_humidity',         'relative humidity (proportion)',      60               ))
unexpanded_col_list.append(ColumnInfo(True, 'recip_rel_humidity',   'reciprocal lative humidity',          60               ))
unexpanded_col_list.append(ColumnInfo(True, 'buoyancy',             'Beucler buoyancy metric',             60               ))
unexpanded_col_list.append(ColumnInfo(True, 'up_integ_tot_cloud',   'ground-up integral of total cloud',   60               ))
unexpanded_col_list.append(ColumnInfo(True, 'down_integ_tot_cloud', 'sky-down integral of total cloud',    60               ))
unexpanded_col_list.append(ColumnInfo(True, 'vert_insolation',      'zenith-adjusted insolation',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_sw_absorb',     'direct shortwave absorbance',          1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_sw_absorb',    'diffuse shortwave absorbance',         1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'direct_lw_absorb',     'direct longwave absorbance',           1, 'W/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'diffuse_lw_absorb',    'diffuse longwave absorbance',          1, 'W/m2'       ))

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


mx_sample_np = mx_sample_np.reshape((num_batches, len(unexpanded_input_col_names)))
sx_sample_np = sx_sample_np.reshape((num_batches, len(unexpanded_input_col_names)))

df_mx = pl.DataFrame(mx_sample_np, schema=unexpanded_input_col_names)
df_sx = pl.DataFrame(sx_sample_np, schema=unexpanded_input_col_names)
df_sy = pl.DataFrame(sx_sample_np, schema=unexpanded_input_col_names)

df_mx.write_csv('batch_mx.csv')
df_sx.write_csv('batch_sx.csv')
df_sy.write_csv('batch_sy.csv')

pass
