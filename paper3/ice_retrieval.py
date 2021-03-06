"""Retrieve ice and dust optical depths from a specified IUVS data file.

This is structured like:
- All the things that are constant in an observation
- All things that are constant in a pixel
- All the things that change each iteration of the solver
"""
# Built-in imports
import glob
import os
import time
from tempfile import mkdtemp
import multiprocessing as mp

# 3rd-party imports
from astropy.io import fits
import numpy as np
from scipy import optimize
import disort
from pyrt.observation import constant_width, phase_to_angles
from pyrt.eos import Hydrostatic
from pyrt.rayleigh import RayleighCO2
from pyrt.aerosol import ForwardScattering, OpticalDepth, \
    TabularLegendreCoefficients
from pyrt.atmosphere import Atmosphere
from pyrt.controller import ComputationalParameters, ModelBehavior
from pyrt.radiation import IncidentFlux, ThermalEmission
from pyrt.output import OutputArrays, OutputBehavior, UserLevel
from pyrt.surface import Surface

# Local imports
from paper3.txt_data import L1CTxt, L2Txt
from paper3.gcm import load_simulation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load in the flatfield
ff = np.load(os.path.join(os.path.dirname(__file__), 'mvn_iuv_flatfield.npy'), allow_pickle=True).item(0)  # 133 bins
flatfield = ff['flatfield']
wavelengths = ff['wavelengths'] / 1000  # convert to microns

# Choose the files I want to retrieve here. Designed for Ubuntu
l1c_files = sorted(glob.glob('/media/kyle/Samsung_T5/l1ctxt/orbit03400/*3453*'))
l2_files = sorted(glob.glob('/media/kyle/Samsung_T5/l2txt/orbit03400/*3453*'))
l1c_file = L1CTxt(l1c_files[5])
l2_file = L2Txt(l2_files[5])

# Flatfield correct
l1c_file.reflectance /= flatfield

# Define spectral info
bin_width = 0.00281   # microns
spectral = constant_width(wavelengths, bin_width)

# HACK: compute all angles at once even though some SZA are on the nightside
l1c_file.solar_zenith_angle = np.where(l1c_file.solar_zenith_angle >= 90, 90, l1c_file.solar_zenith_angle)

# Define angles
angles = phase_to_angles(l1c_file.solar_zenith_angle, l1c_file.emission_angle, l1c_file.phase_angle)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load whatever files possible
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
z_boundaries = np.linspace(80, 0, num=15)  # Define the boundaries to use
gcm_simulation = load_simulation(l1c_file, '/media/kyle/Samsung_T5')
pinterp = gcm_simulation.make_4d_interpolator(gcm_simulation.pressure)
tinterp = gcm_simulation.make_4d_interpolator(gcm_simulation.temperature)
dinterp = gcm_simulation.make_4d_interpolator(gcm_simulation.aerosol)

# TODO: load in dust fsp pf files here
# Dust forward scattering
fsphdul = fits.open('/home/kyle/repos/iuvs_ice/aux/dust_properties.fits')
dust_fsp = fsphdul['primary'].data[:, :, :2]    # Slice off g
dust_fsp_psizes = fsphdul['particle_sizes'].data
dust_fsp_wavs = fsphdul['wavelengths'].data

# Dust phase function
pfhdul = fits.open('/home/kyle/repos/iuvs_ice/aux/dust_phase_function.fits')
dust_phase = pfhdul['primary'].data
dust_pf_psizes = pfhdul['particle_sizes'].data
dust_pf_wavs = pfhdul['wavelengths'].data

# All ice properties
ice_fsp = np.load('/home/kyle/repos/iuvs_ice/aux/ice_forwardscat.npy')
ice_phase = np.load('/home/kyle/repos/iuvs_ice/aux/ice_phase.npy')
ice_phase = np.moveaxis(ice_phase, -1, 0)
ice_psizes = np.load('/home/kyle/repos/iuvs_ice/aux/ice_psize.npy')
# ice_wavs = np.load('/home/kyle/repos/iuvs_ice/aux/ice_wavs.npy')
ice_wavs = ice_fsp[0, :, 0]   # The file was wrong

# Set the particle size profile
dust_pgrad = np.linspace(1.3, 1.8, num=14)
ice_pgrad = np.linspace(2, 2, num=14)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Miscellaneous variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
cp = ComputationalParameters(len(z_boundaries) - 1, 65, 16, 1, 1, 80)
mb = ModelBehavior()
flux = IncidentFlux()
te = ThermalEmission()
ob = OutputBehavior()
ulv = UserLevel(cp.n_user_levels)


def retrieve_pixel(integration: int, position: int):
    # TODO: add altitude != 0 check
    if l1c_file.solar_zenith_angle[integration, position] >= 72 or \
       l1c_file.emission_angle[integration, position] >= 72:
        answer = np.zeros((2, 19)) * np.nan
        print('skipping this due to SZA or EA')
        return position, integration, answer

    # Get the pixel altitude.
    # Suppose the min alt is 18km. This will mean that z boundaries is relative to
    # the surface, *not* the aeroid. If it's relative to the aeroid, I had a bunch of crap that
    # scattered light out of the beam and gave garbage.
    min_alt = gcm_simulation.get_min_alt_at_coord(l1c_file.latitude[integration, position, 4], l1c_file.longitude[integration, position, 4])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make the equation of state variables on a custom grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    inp = []
    for i in range(len(z_boundaries)):
        inp.append([l1c_file.local_time[integration, position],
                    z_boundaries[i] + min_alt,
                    l1c_file.latitude[integration, position, 4],
                    l1c_file.longitude[integration, position, 4]])

    inp = np.array(inp)
    pressure = pinterp(inp)
    temperature = tinterp(inp)
    pressure = np.where(pressure < 0, pressure[-2], pressure)
    temperature = np.where(temperature < 0, temperature[-2], temperature)
    # RegularGridInterpolator *precludes* the need for the original
    # altitudes since it just interpolates them onto the new alt grid.

    # If I do altitude relative to the surface and not the aeroid, these should be unnecessary
    #pressure = np.where(pressure < 0, 0.000001, pressure)
    #temperature = np.where(temperature < 0, 0.000001, temperature)
    mass = 7.3 * 10**-26
    gravity = 3.7

    hydro = Hydrostatic(z_boundaries, pressure, temperature,
                        z_boundaries, mass, gravity)

    ###########################
    # Aerosol things
    ###########################
    # Rayleigh scattering
    rco2 = RayleighCO2(wavelengths, hydro.column_density)
    rayleigh_info = (rco2.optical_depth, rco2.single_scattering_albedo,
                     rco2.phase_function)

    # Set the dust vertical profile
    dust_profile = dinterp(inp)[1:]
    dust_profile = np.where(dust_profile < 0, dust_profile[-2], dust_profile)
    # Set the ice vertical profile
    exp_prof = np.exp(-(z_boundaries + min_alt) / 10)   # 10km scale height
    junk = np.where(z_boundaries < 20, 0, exp_prof)
    ice_prof = (junk[1:] + junk[:-1]) / 2

    ###########################
    # Surface setup
    ###########################
    # Use Todd Clancy's surface
    # I do this on a per pixel bases cause I'm not sure if changing a
    # Surface to Hapke will cause problems.
    clancy_lamber = np.interp(wavelengths, np.linspace(0.2, 0.33, num=100),
                              np.linspace(0.01, 0.015, num=100))
    lamb = [Surface(w, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                ob.only_fluxes) for w in clancy_lamber]

    oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)

    # Make a Hapke surface
    for index, l in enumerate(lamb):
        wolff_hapke = np.interp(wavelengths,
                                np.linspace(0.258, 0.32, num=100),
                                np.linspace(0.07, 0.095, num=100))
        l.make_hapkeHG2_roughness(0.8, 0.06, wolff_hapke[index], 0.3,
                                  0.45, 20,
                                  angles.mu[position, integration],
                                  angles.mu0[position, integration],
                                  angles.phi[position, integration],
                                  angles.phi0[position, integration],
                                  flux.beam_flux)

    def fit_tau(guess: np.ndarray, wav_index: int) -> float:
        dust_guess = guess[0]
        ice_guess = guess[1]

        # Make the dust FSP and PF
        dust_fs = ForwardScattering(dust_fsp[:, :, 1], dust_fsp[:, :, 0], dust_fsp_psizes, dust_fsp_wavs, dust_pgrad, wavelengths, wavelengths[wav_index])
        dust_fs.make_nn_properties()
        dust_od = OpticalDepth(dust_profile, hydro.column_density, dust_fs.extinction, dust_guess)
        dust_tlc = TabularLegendreCoefficients(dust_phase, dust_pf_psizes, dust_pf_wavs, dust_pgrad, wavelengths)
        dust_tlc.make_nn_phase_function()
        dust_info = (dust_od.total, dust_fs.single_scattering_albedo, dust_tlc.phase_function)

        # Make the ice FSP and PF
        ice_fs = ForwardScattering(ice_fsp[:, :, 2], ice_fsp[:, :, 1], ice_psizes, ice_wavs, ice_pgrad, wavelengths, wavelengths[wav_index])
        ice_fs.make_nn_properties()
        ice_od = OpticalDepth(ice_prof, hydro.column_density, ice_fs.extinction, ice_guess)
        ice_tlc = TabularLegendreCoefficients(ice_phase, ice_psizes, ice_wavs, ice_pgrad, wavelengths)
        ice_tlc.make_nn_phase_function()
        ice_info = (ice_od.total, ice_fs.single_scattering_albedo, ice_tlc.phase_function)

        # Put it all together
        model = Atmosphere(rayleigh_info, dust_info, ice_info)

        #print(f'od={model.optical_depth}')
        #print(f'ssa={model.single_scattering_albedo}')
        #print(f'pf={model.legendre_moments}')


        rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
            disort.disort(ob.user_angles, ob.user_optical_depths,
                          ob.incidence_beam_conditions, ob.only_fluxes,
                          mb.print_variables, te.thermal_emission,
                          lamb[wav_index].lambertian,
                          mb.delta_m_plus, mb.do_pseudo_sphere,
                          model.optical_depth[:, wav_index],
                          model.single_scattering_albedo[:, wav_index],
                          model.legendre_moments[:, :, wav_index],
                          hydro.temperature, spectral.low_wavenumber,
                          spectral.high_wavenumber,
                          ulv.optical_depth_output,
                          angles.mu0[position, integration],
                          angles.phi0[position, integration],
                          angles.mu[position, integration],
                          angles.phi[position, integration],
                          flux.beam_flux, flux.isotropic_flux,
                          lamb[wav_index].albedo, te.bottom_temperature,
                          te.top_temperature, te.top_emissivity,
                          mb.radius, hydro.scale_height,
                          lamb[wav_index].rhoq,
                          lamb[wav_index].rhou,
                          lamb[wav_index].rho_accurate,
                          lamb[wav_index].bemst, lamb[wav_index].emust,
                          mb.accuracy, mb.header, oa.direct_beam_flux,
                          oa.diffuse_down_flux, oa.diffuse_up_flux,
                          oa.flux_divergence, oa.mean_intensity,
                          oa.intensity,
                          oa.albedo_medium, oa.transmissivity_medium)
        return (uu[0, 0, 0] - l1c_file.reflectance[integration, position, wav_index]) ** 2

    # Make array to hold the best fit solution
    answer = np.zeros((2, 19))

    for wavelength_index in range(19):
        # skip the ozone wavelengths
        if 5 <= wavelength_index <= 14:
            continue

        # Guess 0.25 for tau dust and 1 for tau ice
        fitted_taus = optimize.minimize(fit_tau, np.array([0.25, 1]),
                                       args=(wavelength_index,), method='Nelder-Mead', bounds=((0, 3), (0, 5))).x
        print(fitted_taus)
        answer[:, wavelength_index] = fitted_taus

    return integration, position, answer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a shared array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
memmap_filename_dust = os.path.join(mkdtemp(), 'myNewFileDust.dat')
memmap_filename_ice = os.path.join(mkdtemp(), 'myNewFileIce.dat')
retrieved_dust = np.memmap(memmap_filename_dust, dtype=float,
                           shape=l1c_file.reflectance.shape, mode='w+')
retrieved_ice = np.memmap(memmap_filename_ice, dtype=float,
                           shape=l1c_file.reflectance.shape, mode='w+')


def make_answer(inp):
    integration = inp[0]
    position = inp[1]
    answer = inp[2]
    retrieved_dust[integration, position, :] = answer[0, :]
    retrieved_ice[integration, position, :] = answer[1, :]


t0 = time.time()
n_cpus = mp.cpu_count()    # = 8 for my desktop, 12 for my laptop
pool = mp.Pool(7)   # save one just to be safe. Some say it's faster

# NOTE: if there are any issues in the argument of apply_async (here,
# retrieve_ssa), it'll break out of that and move on to the next iteration.
#for integ in range(l1c_file.n_integrations):
for integ in range(120):  #range(l1c_file.n_integrations):
    for posit in range(l1c_file.n_positions):
        pool.apply_async(retrieve_pixel, args=(integ, posit), callback=make_answer)
# https://www.machinelearningplus.com/python/parallel-processing-python/
pool.close()
pool.join()  # I guess this postpones further code execution until the queue is finished
np.save('/home/kyle/cloud_retrievals/dust_test.npy', retrieved_dust)
np.save('/home/kyle/cloud_retrievals/ice_test.npy', retrieved_ice)
t1 = time.time()
print(t1-t0)    # It takes 17.6 hours for 133 positions, 170 integrations
