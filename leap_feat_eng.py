# LEAP competition with feature engineering

# This block will be differen in Kaggle notebook:
run_local = True



debug = True
do_test = True

if debug:
    max_train_rows = 2
    max_test_rows = 2
else:
    max_train_rows = 0 # all
    max_test_rows = 0

import copy
import os
import polars as pl


if run_local:
    base_path = '.'
    train_path = os.path.join(base_path, 'train-top.csv')
    test_path = os.path.join(base_path, 'test-top.csv')
    submission_path = os.path.join(base_path, 'sample_submission-top.csv')
else:
    base_path = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
    train_path = os.path.join(base_path, 'train.csv')
    train_path = os.path.join(base_path, 'test.csv')
    submission_path = os.path.join(base_path, 'sample_submission.csv')

# Read in training data
train_df = pl.read_csv(train_path, n_rows=max_train_rows)

# And test/submission 'sample' (weightings)
if do_test:
    test_df = pl.read_csv(test_path, n_rows=max_test_rows)
    submission_df = pl.read_csv(submission_path, n_rows=max_test_rows)

# Altitude levels in hPa from ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc
level_pressure_hpa = [0.07834781133863082, 0.1411083184744011, 0.2529232969453412, 0.4492506351686618, 0.7863461614709879, 1.3473557602677517, 2.244777286900205, 3.6164314830257718, 5.615836425337344, 8.403253219853443, 12.144489352066294, 17.016828024303006, 23.21079811610005, 30.914346261995327, 40.277580662953575, 51.37463234765765, 64.18922841394662, 78.63965761131159, 94.63009200213703, 112.09127353988006, 130.97780378937776, 151.22131809551237, 172.67390465199267, 195.08770981962772, 218.15593476138105, 241.60037901222947, 265.2585152868483, 289.12232222921756, 313.31208711045167, 338.0069992368819, 363.37349177951705, 389.5233382784413, 416.5079218282233, 444.3314120123719, 472.9572063769364, 502.2919169181905, 532.1522731583445, 562.2393924639011, 592.1492760575118, 621.4328411158061, 649.689897132655, 676.6564846051039, 702.2421877859194, 726.4985894989197, 749.5376452869328, 771.4452171682528, 792.2342599534793, 811.8566751313328, 830.2596431972574, 847.4506530638328, 863.5359020075301, 878.7158746040692, 893.2460179738746, 907.3852125876941, 921.3543974831824, 935.3167171670306, 949.3780562075774, 963.5995994020714, 978.013432382012, 992.6355435925217]
num_levels = len(level_pressure_hpa)


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
unexpanded_col_list.append(ColumnInfo(True, 'density',     'air density',                         60, 'kg/m3'      ))
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

expanded_names = [col.name for col in expanded_col_list]
expanded_names_input = [col.name for col in expanded_col_list if col.is_input]
expanded_names_output = [col.name for col in expanded_col_list if not col.is_input]

# Not sure if adding single columns at a time will be too slow on big dataset, can do:
#new_pressure_cols = [pl.lit(level_pressure_hpa[i]).alias(f'pressure_{i}') for i in range(num_levels)]
#train_df = train_df.with_columns(new_pressure_cols)


def add_input_features(df):
    R_air = 287.0 # Mass-based gas constant approx for air in J/kg.K
    for i in range(num_levels):
        # Column names for this level
        cn_pressure       = f'pressure_{i}'   # Pressure in hPa
        cn_temperature    = f'state_t_{i}'    # Temperature in K
        cn_density        = f'density_{i}'    # Density in kg/m3
        cn_mtm_zonal      = f'momentum_u_{i}' # Zonal (E-W) momentum per unit volume in kg/m3.m/s
        cn_mtm_meridional = f'momentum_v_{i}' # Meridional (N-S) momentum per unit volume in kg/m3.m/s
        cn_vel_zonal      = f'state_u_{i}'    # Zonal velocity in m/s
        cn_vel_meridional = f'state_v_{i}'    # Meridional velocity in m/s

        # Using fixed pressure levels, hopefully near enough, not sure in dataset whether
        # we're supposed to scale with surface pressure or something:
        df = df.with_columns(pl.lit(level_pressure_hpa[i]).alias(cn_pressure))
        # pV = mRT
        # m/V = p/RT = density, with *100 for hPa -> Pa conversion
        df = df.with_columns((pl.col(cn_pressure) * 100.0 / (R_air * pl.col(cn_temperature))).alias(cn_density))
        # Momentum per unit vol just density * velocity
        df = df.with_columns((pl.col(cn_density) * pl.col(cn_vel_zonal)).alias(cn_mtm_zonal))
        df = df.with_columns((pl.col(cn_density) * pl.col(cn_vel_meridional)).alias(cn_mtm_meridional))

    return df

train_df = add_input_features(train_df)
if do_test:
    test_df = add_input_features(test_df)

train_df.write_csv('debug_train.csv')
if do_test: test_df.write_csv('debug_test.csv')


# Now trying same as public notebook but with the new features:
# https://www.kaggle.com/code/airazusta014/pytorch-nn/notebook

import random, sys, gc, warnings, math
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from time import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
t0 = time()
np.random.seed(42)
random.seed(42)
min_std = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUGGING = not do_test

#

# load train data
n_rows = max_train_rows
df = train_df
x = df[expanded_names_input].to_numpy().astype(np.float32)
y = df[expanded_names_output].to_numpy().astype(np.float32)

#

# read test
if not DEBUGGING:
    xt = test_df[expanded_names_input].to_numpy().astype(np.float32)
    del test_df
    gc.collect()

#

# norm X
mx = x.mean(axis=0)
sx = np.maximum(x.std(axis=0), min_std)
x = (x - mx.reshape(1,-1)) / sx.reshape(1,-1)
if not DEBUGGING:
    xt = (xt - mx.reshape(1,-1)) / sx.reshape(1,-1)

# norm Y
my = y.mean(axis=0)
sy = np.maximum(np.sqrt((y*y).mean(axis=0)), min_std)
y = (y - my.reshape(1,-1)) / sy.reshape(1,-1)

#

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        
        # Initialize the layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer
            layers.append(nn.LeakyReLU(inplace=True))        # Activation
            layers.append(nn.Dropout(p=0.1))            # Dropout for regularization
            previous_size = hidden_size
        
        # Output layer - no dropout, no activation function
        layers.append(nn.Linear(previous_size, output_size))
        
        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
#

class NumpyDataset(Dataset):
    def __init__(self, x, y):
        """
        Initialize with NumPy arrays.
        """
        assert x.shape[0] == y.shape[0], "Features and labels must have the same number of samples"
        self.x = x
        self.y = y

    def __len__(self):
        """
        Total number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        # Convert the data to tensors when requested
        return torch.from_numpy(self.x[index]).float().to(device), torch.from_numpy(self.y[index]).float().to(device)
    
#

dataset = NumpyDataset(x, y)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 4000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = x.shape[1]
output_size = y.shape[1]
hidden_size = input_size + output_size
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
patience = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    steps = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        # Print every 10 steps
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}')
            total_loss = 0  # Reset the loss for the next 10 steps
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
    for i in range(sy.shape[0]):
        if sy[i] < min_std * 1.1:
            predt[:,i] = 0

    # undo y scaling
    predt = predt * sy.reshape(1,-1) + my.reshape(1,-1)

    for np_col_idx, col_name in enumerate(expanded_names_output):
        submission_df = submission_df.with_columns((pl.col(col_name) * predt[:, np_col_idx]).alias(col_name))
    if debug:
        submission_df.write_csv("submission-debug.csv")
    else:
        submission_df.write_csv("submission.csv")

pass
