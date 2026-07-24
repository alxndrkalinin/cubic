# This code is mostly based on the miplib package (https://github.com/sakoho81/miplib),
# licensed as follows:
#
# Copyright (c) 2018, Sami Koho, Molecular Microscopy & Spectroscopy,
# Italian Institute of Technology. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# * Neither the name of the Molecular Microscopy and Spectroscopy
# research line, nor the names of its contributors may be used to
# endorse or promote products derived from this software without
# specific prior written permission.

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

# In addition to the terms of the license, we ask to acknowledge the use
# of the software in scientific articles by citing:

# Koho, S. et al. Fourier ring correlation simplifies image restoration in fluorescence
# microscopy. Nat. Commun. 10, 3103 (2019).

# Parts of the MIPLIB source code are based on previous BSD licensed
# open source projects:

# pyimagequalityranking:
# Copyright (c) 2015, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# supertomo:
# Copyright (c) 2014, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# iocbio-microscope:
# Copyright (c) 2009-2010, Laboratory of Systems Biology, Institute of
# Cybernetics at Tallinn University of Technology. All rights reserved

"""Utilities for Fourier ring and shell correlation analysis."""

import warnings

import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import UnivariateSpline, interp1d


class FixedDictionary(object):
    """Dictionary with a fixed set of keys.

    Keys are defined at initialization and cannot be added later.
    """

    def __init__(self, keys):
        """Create the dictionary with predefined ``keys``."""
        assert isinstance(keys, (list, tuple))
        self._dictionary = dict.fromkeys(keys)

    def __setitem__(self, key, value):
        """Set ``value`` for ``key`` if it exists."""
        if key not in self._dictionary:
            raise KeyError(f"The key {key} is not defined")
        self._dictionary[key] = value

    def __getitem__(self, key):
        """Return value associated with ``key``."""
        return self._dictionary[key]

    @property
    def keys(self):
        """List of allowed keys."""
        return list(self._dictionary.keys())

    @property
    def contents(self):
        """Return stored keys and values as lists."""
        return list(self._dictionary.keys()), list(self._dictionary.values())


def safe_divide(numerator, denominator):
    """Divide arrays while ignoring divide-by-zero errors."""
    # NaNs are coerced to zero and division warnings are suppressed
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result[result == np.inf] = 0.0
        return np.nan_to_num(result)


class FourierCorrelationDataCollection(object):
    """Container for directional Fourier correlation data."""

    def __init__(self):
        """Create an empty collection."""
        self._data = {}

    def __setitem__(self, key, value):
        """Store ``value`` under integer ``key``."""
        assert isinstance(key, (int, np.integer))
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, key):
        """Return data associated with ``key``."""
        return self._data[str(key)]

    def __iter__(self):
        """Return a fresh iterator over the stored ``(key, value)`` pairs.

        Safe to use while :meth:`FourierCorrelationAnalysis.execute` reassigns
        datasets mid-loop: ``__setitem__`` only overwrites existing keys, so the
        dict never changes size during iteration.
        """
        return iter(self._data.items())

    def __len__(self):
        """Return number of stored datasets."""
        return len(self._data)

    def clear(self):
        """Remove all datasets from the collection."""
        self._data.clear()

    def items(self):
        """Return a list of ``(key, value)`` pairs."""
        return list(self._data.items())


class FourierCorrelationData(object):
    """Container for Fourier correlation data."""

    def __init__(self, data=None):
        """Initialise the data structure with optional ``data`` mapping."""
        correlation_keys = (
            "correlation frequency points-x-bin curve-fit "
            "curve-fit-coefficients correlation-std"
        )
        resolution_keys = (
            "threshold criterion resolution-point "
            "resolution-threshold-coefficients resolution spacing resolution-std"
        )

        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

        if data is not None:
            assert isinstance(data, dict)

            for key, value in data.items():
                if key in self.resolution.keys:
                    self.resolution[key] = value
                elif key in self.correlation.keys:
                    self.correlation[key] = value
                else:
                    raise ValueError("Unknown key found in the initialization data")


def fit_frc_curve(data_set, degree, fit_type="spline", smoothing_factor=0.05):
    """Return curve-fitting function for the provided FRC data."""
    assert isinstance(data_set, FourierCorrelationData)

    data = data_set.correlation["correlation"]

    if fit_type == "smooth-spline":
        equation = UnivariateSpline(data_set.correlation["frequency"], data)
        equation.set_smoothing_factor(smoothing_factor)

    elif fit_type == "spline":
        equation = interp1d(
            data_set.correlation["frequency"],
            data,
            kind="slinear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    elif fit_type == "polynomial":
        coeff = np.polyfit(
            data_set.correlation["frequency"],
            data,
            degree,
            w=1 - data_set.correlation["frequency"] ** 3,
        )
        equation = np.poly1d(coeff)
    else:
        raise ValueError(
            f"Unknown fit_type {fit_type!r}; expected 'smooth-spline', "
            "'spline' or 'polynomial'"
        )

    data_set.correlation["curve-fit"] = equation(data_set.correlation["frequency"])

    return equation


def calculate_snr_threshold_value(points_x_bin, snr):
    """Return the SNR-based resolution threshold curve."""
    nominator = snr + safe_divide(2.0 * np.sqrt(snr) + 1, np.sqrt(points_x_bin))
    denominator = snr + 1 + safe_divide(2.0 * np.sqrt(snr), np.sqrt(points_x_bin))
    return safe_divide(nominator, denominator)


def calculate_resolution_threshold_curve(data_set, criterion, threshold, snr):
    """Compute resolution threshold curve for a given criterion."""
    assert isinstance(data_set, FourierCorrelationData)

    # Copy: patching the empty trailing bin below must not corrupt the stored
    # FRC data as a side effect of computing a threshold curve.
    points_x_bin = np.array(data_set.correlation["points-x-bin"], copy=True)

    if points_x_bin[-1] == 0:
        points_x_bin[-1] = points_x_bin[-2]

    if criterion == "one-bit":
        nominator = 0.5 + safe_divide(2.4142, np.sqrt(points_x_bin))
        denominator = 1.5 + safe_divide(1.4142, np.sqrt(points_x_bin))
        points = safe_divide(nominator, denominator)

    elif criterion == "half-bit":
        nominator = 0.2071 + safe_divide(1.9102, np.sqrt(points_x_bin))
        denominator = 1.2071 + safe_divide(0.9102, np.sqrt(points_x_bin))
        points = safe_divide(nominator, denominator)

    elif criterion == "three-sigma":
        points = safe_divide(
            np.full(points_x_bin.shape, 3.0), (np.sqrt(points_x_bin) + 3.0 - 1)
        )

    elif criterion == "fixed":
        points = np.full(points_x_bin.shape, threshold)
    elif criterion == "snr":
        points = calculate_snr_threshold_value(points_x_bin, snr)

    else:
        raise ValueError(
            f"Unknown resolution threshold criterion {criterion!r}; expected "
            "'one-bit', 'half-bit', 'three-sigma', 'fixed' or 'snr'"
        )

    if criterion != "fixed":
        equation = interp1d(
            data_set.correlation["frequency"],
            points,
            kind="slinear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        curve = equation(data_set.correlation["frequency"])
    else:
        curve = points
        equation = None

    data_set.resolution["threshold"] = curve
    return equation


class FourierCorrelationAnalysis(object):
    """Perform resolution analysis on a collection of FRC data."""

    def __init__(
        self,
        data: FourierCorrelationDataCollection,
        spacing: float,
        *,
        resolution_threshold: str = "fixed",
        threshold_value: float = 0.143,
        snr_value: float = 7.0,
        curve_fit_type: str = "spline",
        curve_fit_degree: int = 3,
        smoothing_factor: float = 0.05,
        verbose: bool = False,
    ) -> None:
        """Store configuration for subsequent analysis."""
        assert isinstance(data, FourierCorrelationDataCollection)

        self.data_collection = data
        self.spacing = spacing
        self.resolution_threshold = resolution_threshold
        self.threshold_value = threshold_value
        self.snr_value = snr_value
        self.curve_fit_type = curve_fit_type
        self.curve_fit_degree = curve_fit_degree
        self.smoothing_factor = smoothing_factor
        self.verbose = verbose

    def execute(self, z_correction=1):
        """Calculate spatial resolution for all datasets."""
        criterion = self.resolution_threshold
        threshold = self.threshold_value
        snr = self.snr_value
        degree = self.curve_fit_degree
        fit_type = self.curve_fit_type
        verbose = self.verbose

        for key, data_set in self.data_collection:
            self.data_collection[int(key)] = self._process_dataset(
                key,
                data_set,
                degree,
                fit_type,
                criterion,
                threshold,
                snr,
                z_correction,
                verbose,
                self.smoothing_factor,
            )

        return self.data_collection

    def _no_resolution(self, data_set, criterion):
        """Mark *data_set* as having no measurable resolution (NaN, not inf)."""
        data_set.resolution["resolution-point"] = (np.nan, np.nan)
        data_set.resolution["criterion"] = criterion
        data_set.resolution["resolution"] = np.nan
        data_set.resolution["spacing"] = self.spacing
        return data_set

    def _process_dataset(
        self,
        key,
        data_set,
        degree,
        fit_type,
        criterion,
        threshold,
        snr,
        z_correction,
        verbose,
        smoothing_factor=0.05,
    ):
        """Process a single dataset and return updated data."""
        if verbose:
            print(f"Calculating resolution point for dataset {key}")

        # Every fit type needs at least 4 points (a cubic spline or polynomial
        # needs m > k). Fewer means the data has almost no populated frequency
        # bins — a strongly anisotropic slice whose min-Nyquist circle covers
        # little of the sampled k-plane, for instance. There is no resolution to
        # measure, but it is worth saying so rather than letting FITPACK raise.
        n_bins = len(data_set.correlation["frequency"])
        if n_bins < 4:
            warnings.warn(
                f"Dataset {key} has only {n_bins} populated frequency bin(s); "
                "too few to fit a resolution curve. Reporting NaN — check the "
                "spacing and shape, or widen bin_delta.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._no_resolution(data_set, criterion)

        frc_eq = fit_frc_curve(data_set, degree, fit_type, smoothing_factor)
        two_sigma_eq = calculate_resolution_threshold_curve(
            data_set, criterion, threshold, snr
        )

        def pdiff1(x):
            return abs(frc_eq(x) - two_sigma_eq(x))

        def pdiff2(x):
            return abs(frc_eq(x) - threshold)

        def first_guess(x, y, thr):
            # Returns ``None`` when no meaningful seed exists. The caller
            # interprets that as "no measurable resolution" and sets
            # ``resolution = NaN``.
            difference = y - thr
            crossings = np.where(difference <= 0)[0]
            if len(crossings) == 0:
                # Never crosses the threshold — resolution is beyond Nyquist
                # (or the prediction is so close to GT that FSC stays above
                # threshold everywhere). No measurable resolution.
                return None
            if crossings[0] == 0:
                # Curve starts already below threshold at the lowest measured
                # frequency — typical of predictions with no meaningful
                # correlation to GT. There is no above-to-below crossing to
                # report; fmin started here would wander into extrapolated
                # territory and yield absurd roots (e.g. negative or near-zero,
                # producing million-µm resolutions).
                return None
            return x[crossings[0] - 1]

        freqs = data_set.correlation["frequency"]
        fit_start = first_guess(
            freqs,
            data_set.correlation["curve-fit"],
            np.mean(data_set.resolution["threshold"]),
        )

        # Handle case where correlation never crosses threshold
        if fit_start is None:
            return self._no_resolution(data_set, criterion)

        if verbose:
            print(f"Fit starts at {fit_start}")
            disp = 1
        else:
            disp = 0
        root = optimize.fmin(
            pdiff2 if criterion == "fixed" else pdiff1, fit_start, disp=disp
        )[0]

        # fmin is unbounded; reject roots that landed outside the measured
        # frequency range (extrapolation territory where the spline can
        # produce spurious threshold crossings). Resolution = 2*spacing/root
        # blows up for root ≈ 0 and is meaningless for root < 0 or root > 1.
        if not (freqs[0] <= root <= freqs[-1]):
            return self._no_resolution(data_set, criterion)

        data_set.resolution["resolution-point"] = (frc_eq(root), root)
        data_set.resolution["criterion"] = criterion

        angle = np.deg2rad(int(key))
        # k(θ) correction from Koho et al. 2019, equation (5)
        # Paper defines θ from XY plane, but our convention is polar angle from Z axis
        # So we use cos(θ) instead of sin(θ) to get:
        #   - θ=0° (Z axis): cos(0)=1 → maximum correction (k=z_correction)
        #   - θ=90° (XY plane): cos(90)=0 → no correction (k=1)
        z_multiplier = 1 + (z_correction - 1) * np.abs(np.cos(angle))
        resolution = z_multiplier * (2 * self.spacing / root)

        data_set.resolution["resolution"] = resolution
        data_set.resolution["spacing"] = self.spacing * z_multiplier

        return data_set
