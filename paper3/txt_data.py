"""This module has tools to create more usable files from ascii files."""
import linecache
import numpy as np


class L1CTxt:
    """A class that can extract all data from an IUVS level 1c txt file.

    It extracts the various properties from an IUVS l1c text file and stores
    them as numpy arrays. It acts somewhat analogously to a .fits file.

    """
    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path
            Absolute path to the IUVS .txt file.

        """
        self.file_path = file_path
        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self._extract_observation_shape_from_file()
        self._pixel_info, self._spectral_info = self._extract_pixel_data()

        self.latitude = self._make_latitude()
        self.longitude = self._make_longitude()
        self.tangent_altitude = self._make_tangent_altitude()
        self.local_time = self._make_local_time()
        self.solar_zenith_angle = self._make_solar_zenith_angle()
        self.emission_angle = self._make_emission_angle()
        self.phase_angle = self._make_phase_angle()
        self.solar_longitude = self._make_solar_longitude()
        self.reflectance = self._make_reflectance()

        del self._pixel_info
        del self._spectral_info

    def _extract_observation_shape_from_file(self) -> tuple[int, int, int]:
        file_shape_line_num = 4
        file_shape = self._extract_info_from_line(file_shape_line_num, int)
        return file_shape[0], file_shape[1], file_shape[2]

    def _extract_info_from_line(self, line_number: int, dtype: object) \
            -> np.ndarray:
        line = linecache.getline(self.file_path, line_number)
        return np.fromstring(line, dtype=dtype, sep=' ')

    def _extract_pixel_data(self) -> tuple[np.ndarray, np.ndarray]:
        pixel_shape = (self.n_integrations, self.n_positions)
        spectral_shape = pixel_shape + (self.n_wavelengths,)
        pixel_info = np.zeros(pixel_shape, dtype=object)
        spectral_info = np.zeros(spectral_shape, dtype=object)

        for line_number, line in enumerate(open(self.file_path, 'r'), start=1):
            if self._is_file_header_line(line_number):
                continue
            line = self._extract_info_from_line(line_number, float)
            if self._is_pixel_info_line(line_number):
                pix_info = _PixelInfo(line)
                pixel_info[pix_info.integration, pix_info.position] = pix_info
            else:
                spec_info = _SpectralInfo(line)
                wav_index = self._make_wavelength_index(line_number)
                spectral_info[pix_info.integration, pix_info.position,
                              wav_index] = spec_info

        return pixel_info, spectral_info

    @staticmethod
    def _is_file_header_line(line_number: int) -> bool:
        return line_number < 6

    def _is_pixel_info_line(self, line_number: int) -> bool:
        return (line_number - 6) % (self.n_wavelengths + 1) == 0

    def _make_wavelength_index(self, line_number) -> int:
        return (line_number - 6) % (self.n_wavelengths + 1) - 1

    # TODO: Can I fix this code duplication?
    def _make_latitude(self) -> np.ndarray:
        latitude = np.zeros(self._pixel_info.shape + (5,))
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                latitude[i, j, :] = self._pixel_info[i, j].latitude
        return latitude

    def _make_longitude(self):
        longitude = np.zeros(self._pixel_info.shape + (5,))
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                longitude[i, j, :] = self._pixel_info[i, j].longitude
        return longitude

    def _make_tangent_altitude(self):
        tan_alt = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                tan_alt[i, j] = self._pixel_info[i, j].tangent_altitude
        return tan_alt

    def _make_local_time(self):
        local_time = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                local_time[i, j] = self._pixel_info[i, j].local_time
        return local_time

    def _make_solar_zenith_angle(self):
        sza = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                sza[i, j] = self._pixel_info[i, j].solar_zenith_angle
        return sza

    def _make_emission_angle(self):
        emission_angle = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                emission_angle[i, j] = self._pixel_info[i, j].emission_angle
        return emission_angle

    def _make_phase_angle(self):
        phase_angle = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                phase_angle[i, j] = self._pixel_info[i, j].phase_angle
        return phase_angle

    def _make_solar_longitude(self):
        solar_longitude = np.zeros(self._pixel_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                solar_longitude[i, j] = self._pixel_info[i, j].solar_longitude
        return solar_longitude

    def _make_reflectance(self):
        reflectance = np.zeros(self._spectral_info.shape)
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                for k in range(self.n_wavelengths):
                    reflectance[i, j, k] = \
                        self._spectral_info[i, j, k].reflectance
        return reflectance


class L2Txt:
    """A class that can extract all data from an IUVS level 12 txt file.

    It extracts *some* of the various properties from an IUVS l2 text file and
    stores them as numpy arrays. It acts somewhat analogously to a .fits file.

    Parameters
    ----------
    file_path
        Absolute path to the IUVS .txt file.

    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._array = self._make_array()

        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self._extract_observation_shape_from_file()

        self.surface_pressure = self._reshape_array_column(self._array[:, 13])
        self.albedo = self._reshape_array_column(self._array[:, -6])
        self.tau_dust = self._reshape_array_column(self._array[:, -4])
        self.o3 = self._reshape_array_column(self._array[:, -2])

        del self._array

    def _make_array(self):
        return np.genfromtxt(self.file_path, skip_header=8)

    def _extract_observation_shape_from_file(self) -> tuple[int, int, int]:
        file_shape_line_num = 4
        file_shape = self._extract_info_from_line(file_shape_line_num, int)
        return file_shape[0], file_shape[1], file_shape[2]

    def _extract_info_from_line(self, line_number: int, dtype: object) \
            -> np.ndarray:
        line = linecache.getline(self.file_path, line_number)
        return np.fromstring(line, dtype=dtype, sep=' ')

    def _reshape_array_column(self, array):
        return array.reshape((self.n_integrations, self.n_positions), order='F')


class _PixelInfo:
    def __init__(self, pixel_info: np.ndarray):
        self.integration = int(pixel_info[0] - 1)
        self.position = int(pixel_info[1] - 1)
        self.latitude = pixel_info[2:7]
        self.longitude = pixel_info[7:12]
        self.tangent_altitude = pixel_info[12]
        self.local_time = pixel_info[13]
        self.solar_zenith_angle = pixel_info[14]
        self.emission_angle = pixel_info[15]
        self.phase_angle = pixel_info[16]
        self.solar_longitude = pixel_info[17]


class _SpectralInfo:
    def __init__(self, spectral_info: np.ndarray):
        self.wavelength = spectral_info[0]
        self.solar_flux = spectral_info[1]
        self.reflectance = spectral_info[2]
        self.uncertainty = spectral_info[3]
