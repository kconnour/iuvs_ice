"""Find the UV single scattering albedo from the MY34 GDS
"""
# Built-in imports
import glob
import os
import time
from tempfile import mkdtemp
import multiprocessing as mp

# 3rd-party imports
import numpy as np
from scipy import optimize
import disort
from pyrt.observation import constant_width, phase_to_angles
from pyrt.eos import Hydrostatic
from pyrt.rayleigh import RayleighCO2
from pyrt.aerosol import Conrath, ForwardScattering, OpticalDepth, \
    TabularLegendreCoefficients
from pyrt.atmosphere import Atmosphere
from pyrt.controller import ComputationalParameters, ModelBehavior
from pyrt.radiation import IncidentFlux, ThermalEmission
from pyrt.output import OutputArrays, OutputBehavior, UserLevel
from pyrt.surface import Surface

# Local imports
from paper3.txt_data import L1CTxt, L2Txt

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
l1c_file = L1CTxt(l1c_files[0])
l2_file = L2Txt(l2_files[0])

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
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
aux_files_path = '/home/kyle/dustssa/kyle_iter2_radprop'
z_boundaries = np.linspace(80, 0, num=15)    # Define the boundaries to use
eos_file = np.flipud(np.load(os.path.join('/home/kyle/repos/pyuvs-rt/data/marsatm.npy')))



# NEED TO sue Franck's surface pressure from the model
eos_file[:, 1] *= 610 / eos_file[-1, 1]
# Construct the EoS profiles
# Use T profile from Kass et al 2019
eos_file[:, 2] = np.flip(np.array(
    [230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
     230, 230, 230, 230, 230, 228, 226, 224, 222, 220,
     214, 208, 202, 196, 190, 186, 182, 178, 174, 170,
     166, 164, 158, 154, 150, 150, 150, 150, 150, 150,
     150]))
mass = 7.3 * 10**-26
gravity = 3.7

hydro = Hydrostatic(eos_file[:, 0], eos_file[:, 1], eos_file[:, 2],
                    z_boundaries, mass, gravity)

###########################
# Aerosol things
###########################
# Rayleigh scattering
rco2 = RayleighCO2(wavelengths, hydro.column_density)
rayleigh_info = (rco2.optical_depth, rco2.single_scattering_albedo,
                 rco2.phase_function)

# Dust vertical profile
z_midpoint = ((z_boundaries[:-1] + z_boundaries[1:]) / 2)
q0 = 1
H = 10
nu = 0.01

conrath = Conrath(z_midpoint, q0, H, nu)
dust_profile = conrath.profile

# Read in the dust properties
cext = np.load(os.path.join(aux_files_path, 'cext_lut.npy'))
csca = np.load(os.path.join(aux_files_path, 'csca_lut.npy'))
fsp_wavs = np.array([0.23, 0.26, 0.28, 0.3])
fsp_psizes = np.array([1.4, 1.6, 1.8, 2])

# TODO: modify this each run
pgrad = np.linspace(2, 2, num=len(z_boundaries) - 1)
wave_ref = 0.88

# Read in the dust phase function
phsfn = np.load(os.path.join(aux_files_path, 'phsfn_coeff.npy'))
pf_wavs = fsp_wavs
pf_psizes = fsp_psizes

# Use constant properties over the MUV region
# fsp_wavs[1] = 0.439
# pf_wavs[1] = 0.439

# Make the misc things
cp = ComputationalParameters(hydro.n_layers, 65, 16, 1, 1, 80)
mb = ModelBehavior()
flux = IncidentFlux()
te = ThermalEmission()
ob = OutputBehavior()
ulv = UserLevel(cp.n_user_levels)

# Surface treatment
# Use Todd Clancy's surface
clancy_lamber = np.interp(wavelengths, np.linspace(0.2, 0.33, num=100),
                          np.linspace(0.01, 0.015, num=100))
lamb = [Surface(w, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                ob.only_fluxes) for w in clancy_lamber]

# Make a Lambert surface. If I want a Hapke surface, that needs to be done in
# retrieve_ssa
#for l in lamb:
#    l.make_lambertian()


def retrieve_ssa(ssa_guess, pixel_index):
    # Skip the retrieval if the pixel quantities aren't good
    answer = np.zeros(19)
    if pixel_od[pixel_index] < 5 or solar_zenith_angle[pixel_index] > 72 or \
            emission_angle[pixel_index] > 72:
        answer[:] = np.nan
        print(f'skipping pixel {pixel_index}')
        return pixel_index, answer

    # Make a Hapke surface
    for index, l in enumerate(lamb):
        wolff_hapke = np.interp(wavelengths, np.linspace(0.258, 0.32, num=100),
                                np.linspace(0.07, 0.095, num=100))
        l.make_hapkeHG2_roughness(0.8, 0.06, wolff_hapke[index], 0.3, 0.45, 20,
                                  angles.mu[pixel_index],
                                  angles.mu0[pixel_index],
                                  angles.phi[pixel_index],
                                  angles.phi0[pixel_index], flux.beam_flux)

    def fit_ssa(guess, wav_index: int):
        # Trap the guess
        if not 0 <= guess <= 1:
            return 9999999
        test_cext = np.copy(cext)
        test_csca = np.copy(csca)

        # BECAUSE I'm using NN, just set all of the coefficients to be my guess
        # test_csca[:, wav_index] = guess * test_cext[:, wav_index]
        test_csca = guess * test_cext

        fs = ForwardScattering(test_csca, test_cext, fsp_psizes, fsp_wavs,
                               pgrad, wavelengths, wave_ref)
        fs.make_nn_properties()

        od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction,
                          pixel_od[pixel_index])

        tlc = TabularLegendreCoefficients(phsfn, pf_psizes, pf_wavs, pgrad,
                                          wavelengths)
        tlc.make_nn_phase_function()

        dust_info = (od.total, fs.single_scattering_albedo, tlc.phase_function)

        model = Atmosphere(rayleigh_info, dust_info)

        oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)

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
                          spectral.high_wavenumber, ulv.optical_depth_output,
                          angles.mu0[pixel_index], angles.phi0[pixel_index],
                          angles.mu[pixel_index], angles.phi[pixel_index],
                          flux.beam_flux, flux.isotropic_flux,
                          lamb[wav_index].albedo, te.bottom_temperature,
                          te.top_temperature, te.top_emissivity,
                          mb.radius, hydro.scale_height, lamb[wav_index].rhoq,
                          lamb[wav_index].rhou, lamb[wav_index].rho_accurate,
                          lamb[wav_index].bemst, lamb[wav_index].emust,
                          mb.accuracy, mb.header, oa.direct_beam_flux,
                          oa.diffuse_down_flux, oa.diffuse_up_flux,
                          oa.flux_divergence, oa.mean_intensity, oa.intensity,
                          oa.albedo_medium, oa.transmissivity_medium)
        return (uu[0, 0, 0] - reflectance[pixel_index, wav_index])**2

    for wavelength in range(19):
        fitted_ssa = optimize.minimize(fit_ssa, np.array([ssa_guess]),
                                       wavelength, method='Nelder-Mead').x
        print(f'Got an answer of {fitted_ssa}')
        answer[wavelength] = fitted_ssa
    return pixel_index, answer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a shared array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
memmap_filename = os.path.join(mkdtemp(), 'myNewFile.dat')
retrieved_ssas = np.memmap(memmap_filename, dtype=float,
                           shape=reflectance.shape, mode='w+')


def make_answer(inp):
    # inp must match the above: return pixel_index, answer
    retrieved_ssas[inp[0], :] = inp[1]


t0 = time.time()
n_cpus = mp.cpu_count()    # = 8 for my desktop, 12 for my laptop
pool = mp.Pool(7)   # save one just to be safe. Some say it's faster

# NOTE: if there are any issues in the argument of apply_async (here,
# retrieve_ssa), it'll break out of that and move on to the next iteration.
for pixel in range(reflectance.shape[0]):
    pool.apply_async(retrieve_ssa, args=(0.65, pixel), callback=make_answer)
# https://www.machinelearningplus.com/python/parallel-processing-python/
pool.close()
pool.join()  # I guess this postpones further code execution until the queue is finished
np.save('/home/kyle/ssa_retrievals/iteration2/new-fsp_new-pf_hapke-wolff_2-0size.npy', retrieved_ssas)
t1 = time.time()
print(t1-t0)
