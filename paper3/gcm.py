import numpy as np
from pathlib import Path
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator
from paper3.txt_data import L1CTxt


def select_gcm_run_number(ls: float) -> int:
    ls_bin = np.floor(ls / 5)
    return int(ls_bin + 73)


def get_gcm_filename(ls: float) -> str:
    num = select_gcm_run_number(ls)
    return f'stats{num}_A.nc'


class GCMSimulation:
    """Copy the GCM simulations here. Flip the variables to be strictly ascending
    for RegularGridInterpolator

    """
    def __init__(self, simulation_location: str):
        file = netcdf_file(simulation_location)
        self.lat = np.copy(np.flip(file.variables['latitude'][:]))  # (49,)
        self.lon = np.copy(file.variables['longitude'][:])  # (65,)  [-180, +180]
        self.alt = np.copy(file.variables['altitude'][:])  # (111,)
        self.time = np.copy(file.variables['Time'][:])  # (12,)
        self.pressure = np.copy(np.flip(file.variables['pressure'][:], axis=2))  # (12, 111, 49, 65)   units of Pascal, where 1 Pa = 0.01 mbar
        self.temperature = np.copy(np.flip(file.variables['temp'][:], axis=2))  # (12, 111, 49, 65)
        self.aerosol = np.copy(np.flip(file.variables['aerosol'][:], axis=2))  # (12, 111, 49, 65)
        file.close()

    def make_4d_interpolator(self, quantity, method='linear'):
        return RegularGridInterpolator((self.time, self.alt, self.lat, self.lon),
                                       quantity, method=method)


def load_simulation(file: L1CTxt, ssd_path: str):
    ssd = Path(ssd_path)
    filename = get_gcm_filename(file.solar_longitude[0, 0])
    return GCMSimulation(ssd.joinpath('gcm_runs').joinpath(filename))



'''foo = netcdf_file('/media/kyle/Samsung_T5/gcm_runs/stats73_A.nc')

for f in foo.variables:
    print(f)

lat = foo.variables['latitude']         # (49,)
lon = foo.variables['longitude']        # (65,)
alt = foo.variables['altitude']         # (111,)
time = foo.variables['Time']            # (12,)
pressure = foo.variables['pressure']    # (12, 111, 49, 65)
aerosol = foo.variables['aerosol']      # (12, 111, 49, 65)
print(np.amax(aerosol[:]))
foo.close()
'''

if __name__ == '__main__':
    f = '/Volumes/Samsung_T5/gcm_runs/stats73_A.nc'
    g = GCMSimulation(f)
    nd = RegularGridInterpolator((g.time, g.alt, g.lat, g.lon), g.pressure, method='linear')
    inp = np.array([15, 30.234, 10, -25])
    print(nd(inp))
