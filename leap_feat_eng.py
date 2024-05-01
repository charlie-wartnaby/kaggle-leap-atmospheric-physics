# LEAP competition with feature engineering

import copy
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

# Altitude levels in hPa from ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc
level_pressure_hpa = [0.07834781133863082, 0.1411083184744011, 0.2529232969453412, 0.4492506351686618, 0.7863461614709879, 1.3473557602677517, 2.244777286900205, 3.6164314830257718, 5.615836425337344, 8.403253219853443, 12.144489352066294, 17.016828024303006, 23.21079811610005, 30.914346261995327, 40.277580662953575, 51.37463234765765, 64.18922841394662, 78.63965761131159, 94.63009200213703, 112.09127353988006, 130.97780378937776, 151.22131809551237, 172.67390465199267, 195.08770981962772, 218.15593476138105, 241.60037901222947, 265.2585152868483, 289.12232222921756, 313.31208711045167, 338.0069992368819, 363.37349177951705, 389.5233382784413, 416.5079218282233, 444.3314120123719, 472.9572063769364, 502.2919169181905, 532.1522731583445, 562.2393924639011, 592.1492760575118, 621.4328411158061, 649.689897132655, 676.6564846051039, 702.2421877859194, 726.4985894989197, 749.5376452869328, 771.4452171682528, 792.2342599534793, 811.8566751313328, 830.2596431972574, 847.4506530638328, 863.5359020075301, 878.7158746040692, 893.2460179738746, 907.3852125876941, 921.3543974831824, 935.3167171670306, 949.3780562075774, 963.5995994020714, 978.013432382012, 992.6355435925217]


# Manage columns as described in competition
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
    ColumnInfo(True,  'pbuf_COSZRS',      'cosine of solar zenith angle',                         1, 'N/m2'   ),
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

# Add columns for new features
unexpanded_col_list.append(ColumnInfo(True, 'pressure',    'air pressure',                        60, 'N/m2'       ))
unexpanded_col_list.append(ColumnInfo(True, 'air_density', 'air density',                         60, 'kg/m3'      ))
unexpanded_col_list.append(ColumnInfo(True, 'momentum_u',  'zonal momentum per unit volume',      60, '(kg.m/s)/m3')),
unexpanded_col_list.append(ColumnInfo(True, 'momentum_v',  'meridional momentum per unit volume', 60, '(kg.m/s)/m3')),

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

print(f'expanded_col_list len={len(expanded_col_list)}')
expanded_names = [col.name for col in expanded_col_list]
print('Expanded col list names', expanded_names)
