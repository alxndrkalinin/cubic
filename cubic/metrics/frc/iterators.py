# This code is mostly based on the miplib package (https://github.com/sakoho81/miplib),
# licensed as follows:
#
# Copyright (c) 2018, Sami Koho, Molecular Microscopy & Spectroscopy,
# Italian Institute of Technology. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# * Neither the name of the Molecular Microscopy and Spectroscopy
# research line, nor the names of its contributors may be used to
# endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER AND CONTRIBUTORS ''AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Iterator utilities for FRC calculations."""

# mypy: ignore-errors

from math import floor
from collections.abc import Iterable

import numpy as np

from cubic.skimage import exposure
from cubic.image_utils import rotate_image

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def cast_to_dtype(
    data: np.ndarray,
    dtype: np.dtype,
    rescale: bool = True,
    remove_outliers: bool = False,
) -> np.ndarray:
    """Cast ``data`` into ``dtype`` optionally rescaling to the new dynamic range."""
    if data.dtype == dtype:
        return data

    if "int" in str(dtype):
        data_info = np.iinfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    elif "float" in str(dtype):
        data_info = np.finfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    else:
        data_max = data.max()
        data_min = data.min()
        print("Warning casting into unknown data type. Detail clipping may occur")

    # In case of unsigned integers, numbers below zero need to be clipped
    if "uint" in str(dtype):
        data_max = 255
        data_min = 0

    if remove_outliers:
        data = data.clip(0, np.percentile(data, 99.99))

    if rescale:
        return rescale_to_min_max(data, data_min, data_max).astype(dtype)

    return data.clip(data_min, data_max).astype(dtype)


def rescale_to_min_max(
    data: np.ndarray, data_min: float, data_max: float
) -> np.ndarray:
    """Rescale ``data`` intensities to ``[data_min, data_max]``."""
    return exposure.rescale_intensity(data, out_range=(data_min, data_max))


def expand_to_shape(
    data: np.ndarray,
    shape: Iterable[int],
    dtype: np.dtype | None = None,
    background: float | None = None,
) -> np.ndarray:
    """Expand ``data`` to ``shape`` by zero padding."""
    if dtype is None:
        dtype = data.dtype

    start_index = np.array(shape) - data.shape
    data_start = np.negative(start_index.clip(max=0))
    data = cast_to_dtype(data, dtype, rescale=False)
    if data.ndim == 3:
        data = data[data_start[0] :, data_start[1] :, data_start[2] :]
    else:
        data = data[data_start[0] :, data_start[1] :]

    if background is None:
        background = 0

    if tuple(shape) == data.shape:
        return data

    expanded_data = np.zeros(shape, dtype=dtype) + background
    slices = []
    rhs_slices = []
    for s1, s2 in zip(shape, data.shape):
        a, b = (s1 - s2 + 1) // 2, (s1 + s2 + 1) // 2
        c, d = 0, s2
        while a < 0:
            a += 1
            b -= 1
            c += 1
            d -= 1
        slices.append(slice(a, b))
        rhs_slices.append(slice(c, d))
    try:
        expanded_data[tuple(slices)] = data[tuple(rhs_slices)]
    except ValueError as exc:
        print(data.shape, shape)
        raise ValueError("Failed to expand data to the requested shape") from exc
    return expanded_data


def _angle_mask(phi: np.ndarray, phi_min: float, phi_max: float) -> np.ndarray:
    """Return a boolean mask for an angular sector."""
    arr_inf = phi >= phi_min
    arr_sup = phi < phi_max

    arr_inf_neg = phi >= phi_min + np.pi
    arr_sup_neg = phi < phi_max + np.pi

    return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg


# ---------------------------------------------------------------------------
# Iterator classes
# ---------------------------------------------------------------------------


class FourierRingIterator:
    """Iterate over concentric Fourier rings for 2D images (unshifted FFT)."""

    def __init__(self, shape: Iterable[int], d_bin: int) -> None:
        if len(shape) != 2:
            raise AssertionError("shape must be 2D")

        self.d_bin = d_bin
        self.ring_start = 0
        self._nbins = int(floor(shape[0] / (2 * self.d_bin)))
        # Use unshifted fftfreq coordinates (no fftshift)
        axes = [np.fft.fftfreq(n) * n for n in shape]
        y, x = np.meshgrid(*axes, indexing="ij")
        self.meshgrid = (y, x)
        self.r = np.sqrt(x**2 + y**2)
        self.current_ring = self.ring_start
        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        # Use bin midpoints matching histogram backend
        bin_starts = np.arange(0, self.freq_nyq, self.d_bin)
        self._radii = bin_starts + 0.5 * self.d_bin

    @property
    def radii(self) -> np.ndarray:
        """Radii for the concentric rings."""
        return self._radii

    @property
    def nbins(self) -> int:
        """Number of available rings."""
        return self._nbins

    def get_points_on_ring(self, ring_start: int, ring_stop: int) -> np.ndarray:
        """Return boolean mask for points on a ring. DC term (r==0) is excluded."""
        arr_inf = self.r >= ring_start
        arr_sup = self.r < ring_stop
        ring_mask = np.logical_and(arr_inf, arr_sup)
        # Exclude DC term (r==0, with small epsilon for floating point safety)
        eps = 1e-10
        ring_mask = np.logical_and(ring_mask, self.r > eps)
        return ring_mask

    def __iter__(self) -> "FourierRingIterator":
        """Return iterator over rings."""
        return self

    def __next__(self):  # -> tuple[tuple[np.ndarray, np.ndarray], int]
        """Return mask and index for the next ring."""
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring - 1


class SectionedFourierRingIterator(FourierRingIterator):
    """Fourier ring iterator that yields only a specific rotated section."""

    def __init__(self, shape: Iterable[int], d_bin: int, d_angle: int) -> None:
        FourierRingIterator.__init__(self, shape, d_bin)
        self.d_angle = np.deg2rad(d_angle)
        y, x = self.meshgrid
        self.phi = np.arctan2(y, x) + np.pi
        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi
        self._angle = 0
        self.angle_sector = self.get_angle_sector(0, d_bin)

    @property
    def angle(self) -> float:
        """Current rotation angle in radians."""
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        angle = np.deg2rad(value)
        self._angle = angle
        self.angle_sector = self.get_angle_sector(angle, angle + self.d_angle)

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return mask for the angular sector between ``phi_min`` and ``phi_max``."""
        return _angle_mask(self.phi, phi_min, phi_max)

    def __getitem__(self, limits: tuple[int, int, float, float]):
        """Return coordinates for a ring section defined by ``limits``."""
        ring_start, ring_stop, angle_min, angle_max = limits
        ring = self.get_points_on_ring(ring_start, ring_stop)
        cone = self.get_angle_sector(angle_min, angle_max)
        return np.where(ring * cone)

    def __next__(self):
        """Return next ring limited to the selected angular sector."""
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring * self.angle_sector), self.current_ring - 1


class FourierShellIterator:
    """Simple iterator over concentric Fourier shells for 3D images (unshifted FFT)."""

    def __init__(self, shape: Iterable[int], d_bin: int) -> None:
        self.d_bin = d_bin
        # Use unshifted fftfreq coordinates (no fftshift)
        axes = [np.fft.fftfreq(n) * n for n in shape]
        z, y, x = np.meshgrid(*axes, indexing="ij")
        self.meshgrid = (z, y, x)
        self.r = np.sqrt(x**2 + y**2 + z**2)
        self.shell_start = 0
        self.shell_stop = int(floor(shape[0] / (2 * self.d_bin))) - 1
        self.current_shell = self.shell_start
        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        # Use bin midpoints matching histogram backend
        bin_starts = np.arange(0, self.freq_nyq, self.d_bin)
        self.radii = bin_starts + 0.5 * self.d_bin

    @property
    def steps(self) -> np.ndarray:
        """Available shell radii."""
        return self.radii

    @property
    def nyquist(self) -> int:
        """Nyquist frequency for the current shape."""
        return self.freq_nyq

    def get_points_on_shell(self, shell_start: int, shell_stop: int) -> np.ndarray:
        """Return boolean mask for points within a shell. DC term (r==0) is excluded."""
        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop
        shell_mask = arr_inf * arr_sup
        # Exclude DC term (r==0, with small epsilon for floating point safety)
        eps = 1e-10
        shell_mask = shell_mask * (self.r > eps)
        return shell_mask

    def __getitem__(self, limits: tuple[int, int]):
        """Return coordinates for points within ``limits`` shell."""
        shell_start, shell_stop = limits
        shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(shell)

    def __iter__(self) -> "FourierShellIterator":
        """Return iterator over shells."""
        return self

    def __next__(self):
        """Return mask and index for the next shell."""
        shell_idx = self.current_shell
        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_shell += 1
        return np.where(shell), shell_idx


class SectionedFourierShellIterator(FourierShellIterator):
    """Fourier shell iterator that divides each shell into angular sections."""

    def __init__(self, shape: Iterable[int], d_bin: int, d_angle: int) -> None:
        FourierShellIterator.__init__(self, shape, d_bin)
        self.d_angle = np.deg2rad(d_angle)
        z, y, x = self.meshgrid
        self.phi = np.arctan2(y, z) + np.pi
        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi
        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start
        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        """Radii and angles covered by the iterator."""
        return self.radii, self.angles

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return mask for an angular sector of a shell."""
        return _angle_mask(self.phi, phi_min, phi_max)

    def __getitem__(self, limits: tuple[int, int, float, float]):
        """Return coordinates for a shell sector defined by ``limits``."""
        shell_start, shell_stop, angle_min, angle_max = limits
        angle_min = np.deg2rad(angle_min)
        angle_max = np.deg2rad(angle_max)
        shell = self.get_points_on_shell(shell_start, shell_stop)
        cone = self.get_angle_sector(angle_min, angle_max)
        return np.where(shell * cone)

    def __next__(self):
        """Return coordinates for the next shell-angle pair."""
        rotation_idx = self.current_rotation
        shell_idx = self.current_shell
        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
            cone = self.get_angle_sector(
                self.current_rotation * self.d_angle,
                (self.current_rotation + 1) * self.d_angle,
            )
        else:
            raise StopIteration

        if rotation_idx >= self.rotation_stop:
            self.current_rotation = 0
            self.current_shell += 1
        else:
            self.current_rotation += 1

        return np.where(shell * cone), shell_idx, rotation_idx


class HollowSectionedFourierShellIterator(SectionedFourierShellIterator):
    """Sectioned shell iterator with a hollowed central region."""

    def __init__(
        self, shape: Iterable[int], d_bin: int, d_angle: int, d_extract_angle: int = 5
    ) -> None:
        SectionedFourierShellIterator.__init__(self, shape, d_bin, d_angle)
        self.d_extract_angle = np.deg2rad(d_extract_angle)

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return sector mask with a hollowed center."""
        full_section = _angle_mask(self.phi, phi_min, phi_max)
        sector_center = phi_min + (phi_max - phi_min) / 2
        phi_min_ext = sector_center - self.d_extract_angle
        phi_max_ext = sector_center + self.d_extract_angle
        extract_section = _angle_mask(self.phi, phi_min_ext, phi_max_ext)
        return np.logical_xor(full_section, extract_section)


class AxialExcludeSectionedFourierShellIterator(HollowSectionedFourierShellIterator):
    """Sectioned shell iterator that excludes cones around the axial direction."""

    def __init__(
        self, shape: Iterable[int], d_bin: int, d_angle: int, d_extract_angle: int = 5
    ) -> None:
        HollowSectionedFourierShellIterator.__init__(self, shape, d_bin, d_angle)
        self.d_extract_angle = np.deg2rad(d_extract_angle)

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return sector mask excluding regions near the axis."""
        full_section = _angle_mask(self.phi, phi_min, phi_max)
        axis_pos = np.deg2rad(90) + self.d_angle / 2
        axis_neg = np.deg2rad(270) + self.d_angle / 2

        if phi_min <= axis_pos <= phi_max:
            phi_min_ext = axis_pos - self.d_extract_angle
            phi_max_ext = axis_pos + self.d_extract_angle
        elif phi_min <= axis_neg <= phi_max:
            phi_min_ext = axis_neg - self.d_extract_angle
            phi_max_ext = axis_neg + self.d_extract_angle
        else:
            return full_section

        extract_section = _angle_mask(self.phi, phi_min_ext, phi_max_ext)
        return np.logical_xor(full_section, extract_section)


class RotatingFourierShellIterator(FourierShellIterator):
    """Fourier shell iterator that rotates a plane through the volume."""

    def __init__(self, shape: Iterable[int], d_bin: int, d_angle: int) -> None:
        if len(shape) != 3:
            raise AssertionError("This iterator assumes a 3D shape")

        FourierShellIterator.__init__(self, shape, d_bin)
        plane = expand_to_shape(np.ones((1, shape[1], shape[2])), shape)
        self.plane = plane
        self.rotated_plane = plane > 0
        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start
        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        """Radii and rotation angles."""
        return self.radii, self.angles

    def __getitem__(self, limits: tuple[int, int, float]):
        """Return coordinates from a rotated plane within a shell."""
        shell_start, shell_stop, angle = limits
        rotated_plane = rotate_image(self.plane, angle)
        points_on_plane = rotated_plane > 0
        points_on_shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(points_on_plane * points_on_shell)

    def __next__(self):
        """Return coordinates for the next rotated shell."""
        rotation_idx = self.current_rotation + 1
        shell_idx = self.current_shell
        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
            self.current_shell += 1
        elif rotation_idx <= self.rotation_stop:
            rotated_plane = rotate_image(
                self.plane, self.angles[rotation_idx], interpolation="linear"
            )
            self.rotated_plane = rotated_plane > 0
            self.current_shell = 0
            shell_idx = 0
            self.current_rotation += 1
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
        else:
            raise StopIteration

        return np.where(shell * self.rotated_plane), shell_idx, self.current_rotation
