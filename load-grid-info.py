import xarray as xr

grid_path = 'ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc'

grid_info = xr.open_dataset(grid_path)

levels = grid_info['lev']

# Get the 60 levels as pressures in hPa
print(levels.units, levels.values)

pass
