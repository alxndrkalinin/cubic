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

from collections.abc import Iterable, Sequence

import numpy as np

from .radial import _kmax_phys, radial_edges, _spacing_or_unit

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


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

    def __init__(
        self,
        shape: Iterable[int],
        d_bin: int,
        spacing: Sequence[float] | None = None,
        exclude_overflow: bool = False,
    ) -> None:
        if len(shape) != 2:
            raise AssertionError("shape must be 2D")

        shape = tuple(shape)
        self.exclude_overflow = exclude_overflow

        # Use radial_edges for consistent binning with histogram backend
        self.edges, self._radii = radial_edges(shape, d_bin, spacing=spacing)
        self._nbins = len(self._radii)

        # Use unshifted fftfreq coordinates (no fftshift)
        # Match radial_bin_id formula for consistency
        axes = [
            np.fft.fftfreq(n, d=sp)
            for n, sp in zip(shape, _spacing_or_unit(spacing, len(shape)))
        ]
        y, x = np.meshgrid(*axes, indexing="ij")
        self.meshgrid = (y, x)
        self.r = np.sqrt(x**2 + y**2)

        self.current_ring = 0

    @property
    def radii(self) -> np.ndarray:
        """Radii for the concentric rings (bin midpoints)."""
        return self._radii

    @property
    def nbins(self) -> int:
        """Number of available rings."""
        return self._nbins

    def get_points_on_ring(
        self, ring_start: float, ring_stop: float, is_last: bool = False
    ) -> np.ndarray:
        """Return boolean mask for points on a ring. DC term (r==0) is excluded.

        Args:
            ring_start: Lower edge of ring
            ring_stop: Upper edge of ring
            is_last: If True, include all points >= ring_start, folding the
                frequencies between kmax and the FFT corners into this ring
                (see ``exclude_overflow``).
        """
        arr_inf = self.r >= ring_start
        if is_last:
            # Last bin: include all points >= ring_start (overflow bins)
            ring_mask = arr_inf
        else:
            # Regular bin: ring_start <= r < ring_stop
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
            is_last = self.current_ring == self._nbins - 1 and not self.exclude_overflow
            ring = self.get_points_on_ring(
                self.edges[self.current_ring],
                self.edges[self.current_ring + 1],
                is_last=is_last,
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring - 1


class FourierShellIterator:
    """Simple iterator over concentric Fourier shells for 3D images (unshifted FFT)."""

    def __init__(
        self,
        shape: Iterable[int],
        d_bin: int,
        spacing: Sequence[float] | None = None,
        exclude_overflow: bool = False,
    ) -> None:
        shape = tuple(shape)
        self.exclude_overflow = exclude_overflow

        # Use radial_edges for consistent binning with histogram backend
        self.edges, self.radii = radial_edges(shape, d_bin, spacing=spacing)
        self.shell_start = 0
        self.shell_stop = len(self.radii) - 1

        # Use unshifted fftfreq coordinates (no fftshift)
        # Match radial_bin_id formula for consistency
        spacing_units = _spacing_or_unit(spacing, len(shape))
        axes = [np.fft.fftfreq(n, d=sp) for n, sp in zip(shape, spacing_units)]
        z, y, x = np.meshgrid(*axes, indexing="ij")
        self.meshgrid = (z, y, x)
        self.r = np.sqrt(x**2 + y**2 + z**2)

        self.current_shell = self.shell_start
        # Compute Nyquist frequency using same functions as radial_edges
        self.freq_nyq = _kmax_phys(shape, spacing_units)

    @property
    def steps(self) -> np.ndarray:
        """Available shell radii (bin midpoints)."""
        return self.radii

    @property
    def nyquist(self) -> float:
        """Nyquist frequency for the current shape (physical or index units)."""
        return self.freq_nyq

    def get_points_on_shell(
        self, shell_start: float, shell_stop: float, is_last: bool = False
    ) -> np.ndarray:
        """Return boolean mask for points within a shell. DC term (r==0) is excluded.

        Args:
            shell_start: Lower edge of shell
            shell_stop: Upper edge of shell
            is_last: If True, include all points >= shell_start, folding the
                frequencies between kmax and the FFT corners into this shell.
                In 3D that is roughly half of all voxels, so the last value is
                not a shell average (see ``exclude_overflow``).
        """
        arr_inf = self.r >= shell_start
        if is_last:
            # Last bin: include all points >= shell_start (overflow bins)
            shell_mask = arr_inf
        else:
            # Regular bin: shell_start <= r < shell_stop
            arr_sup = self.r < shell_stop
            shell_mask = arr_inf * arr_sup
        # Exclude DC term (r==0, with small epsilon for floating point safety)
        eps = 1e-10
        shell_mask = shell_mask * (self.r > eps)
        return shell_mask

    def __iter__(self) -> "FourierShellIterator":
        """Return iterator over shells."""
        return self

    def __next__(self):
        """Return mask and index for the next shell."""
        shell_idx = self.current_shell
        if shell_idx <= self.shell_stop:
            is_last = shell_idx == self.shell_stop and not self.exclude_overflow
            shell = self.get_points_on_shell(
                self.edges[self.current_shell],
                self.edges[self.current_shell + 1],
                is_last=is_last,
            )
        else:
            raise StopIteration

        self.current_shell += 1
        return np.where(shell), shell_idx


class SectionedFourierShellIterator(FourierShellIterator):
    """Fourier shell iterator that divides each shell into angular sections."""

    def __init__(
        self,
        shape: Iterable[int],
        d_bin: int,
        d_angle: int,
        spacing: Sequence[float] | None = None,
    ) -> None:
        FourierShellIterator.__init__(self, shape, d_bin, spacing=spacing)
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

    def __next__(self):
        """Return coordinates for the next shell-angle pair."""
        rotation_idx = self.current_rotation
        shell_idx = self.current_shell
        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.edges[self.current_shell], self.edges[self.current_shell + 1]
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
        self,
        shape: Iterable[int],
        d_bin: int,
        d_angle: int,
        d_extract_angle: int = 5,
        spacing: Sequence[float] | None = None,
    ) -> None:
        SectionedFourierShellIterator.__init__(
            self, shape, d_bin, d_angle, spacing=spacing
        )
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
        self,
        shape: Iterable[int],
        d_bin: int,
        d_angle: int,
        d_extract_angle: int | float = 5,
        spacing: Sequence[float] | None = None,
    ) -> None:
        HollowSectionedFourierShellIterator.__init__(
            self, shape, d_bin, d_angle, d_extract_angle, spacing=spacing
        )
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
