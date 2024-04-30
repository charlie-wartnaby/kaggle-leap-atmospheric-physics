import xarray as xr

grid_path = 'ClimSim-main\grid_info\ClimSim_low-res_grid-info.nc'

grid_info = xr.open_dataset(grid_path)

levels = grid_info['lev']

# Get the 60 levels as pressures in hPa
print(f"# Altitude levels in {levels.units} from {grid_path}")
string_levels = levels.values.astype(str)
print(f"level_pressure_{levels.units.lower()} = [" + ", ".join(string_levels) + "]")
