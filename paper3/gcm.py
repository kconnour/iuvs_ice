import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import LinearNDInterpolator


def gcm_file_selector(ls: float) -> int:
    ls_bin = np.floor(ls / 5)
    return int(ls_bin + 73)


class GCMSimulation:
    def __init__(self, simulation_location: str):
        file = netcdf_file(simulation_location)
        self.lat = file.variables['latitude'][:]  # (49,)
        self.lon = file.variables['longitude'][:]  # (65,)  [-180, +180]
        self.alt = file.variables['altitude'][:]  # (111,)
        self.time = file.variables['Time'][:]  # (12,)
        self.pressure = file.variables['pressure'][:]  # (12, 111, 49, 65)
        self.aerosol = file.variables['aerosol'][:]  # (12, 111, 49, 65)

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
    f = '/media/kyle/Samsung_T5/gcm_runs/stats73_A.nc'
    g = GCMSimulation(f)
    coords = list(zip(g.time, g.alt, g.lat, g.lon))
    nd = LinearNDInterpolator(coords, g.pressure)
    print(nd(15, 30.234, 10, -25))
