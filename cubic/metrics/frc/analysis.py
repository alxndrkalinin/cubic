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

from argparse import Namespace

import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import UnivariateSpline, interp1d


def get_frc_options(
    bin_delta: int = 1,
    angle_delta: int = 15,
    extract_angle_delta: float = 0.1,
    resolution_threshold: str = "fixed",
    curve_fit_type: str = "spline",
    disable_hamming: bool = False,
    verbose: bool = False,
) -> Namespace:
    """Return a :class:`argparse.Namespace` with common FRC/FSC parameters."""
    return Namespace(
        d_bin=bin_delta,
        d_angle=angle_delta,
        d_extract_angle=extract_angle_delta,
        disable_hamming=disable_hamming,
        resolution_threshold_criterion=resolution_threshold,
        resolution_threshold_value=0.143,
        resolution_snr_value=7.0,
        frc_curve_fit_degree=3,
        frc_curve_fit_type=curve_fit_type,
        verbose=verbose,
    )


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

        self.iter_index = 0

    def __setitem__(self, key, value):
        """Store ``value`` under integer ``key``."""
        assert isinstance(key, (int, np.integer))
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, key):
        """Return data associated with ``key``."""
        return self._data[str(key)]

    def __iter__(self):
        """Return iterator over stored datasets."""
        return self

    def __next__(self):
        """Return next ``(key, value)`` pair."""
        try:
            item = list(self._data.items())[self.iter_index]
        except IndexError:
            self.iter_index = 0
            raise StopIteration

        self.iter_index += 1
        return item

    def __len__(self):
        """Return number of stored datasets."""
        return len(self._data)

    def clear(self):
        """Remove all datasets from the collection."""
        self._data.clear()

    def items(self):
        """Return a list of ``(key, value)`` pairs."""
        return list(self._data.items())

    def nitems(self):
        """Alias for :func:`__len__`."""
        return len(self._data)


class FourierCorrelationData(object):
    """Container for Fourier correlation data."""

    # todo: the dictionary format here is a bit clumsy. Maybe change to a simpler structure

    def __init__(self, data=None):
        """Initialise the data structure with optional ``data`` mapping."""
        correlation_keys = (
            "correlation frequency points-x-bin curve-fit curve-fit-coefficients"
        )
        resolution_keys = (
            "threshold criterion resolution-point "
            "resolution-threshold-coefficients resolution spacing"
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


def fit_frc_curve(data_set, degree, fit_type="spline"):
    """Return curve-fitting function for the provided FRC data."""
    assert isinstance(data_set, FourierCorrelationData)

    data = data_set.correlation["correlation"]

    if fit_type == "smooth-spline":
        equation = UnivariateSpline(data_set.correlation["frequency"], data)
        equation.set_smoothing_factor(0.25)
        # equation = interp1d(data_set.correlation["frequency"],
        #                     data, kind='slinear')

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
        raise AttributeError(fit_type)

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

    points_x_bin = data_set.correlation["points-x-bin"]

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
        points = threshold * np.ones(len(data_set.correlation["points-x-bin"]))
    elif criterion == "snr":
        points = calculate_snr_threshold_value(points_x_bin, snr)

    else:
        raise AttributeError()

    if criterion != "fixed":
        # coeff = np.polyfit(data_set.correlation["frequency"], points, 3)
        # equation = np.poly1d(coeff)
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
        self.verbose = verbose

    def execute(self, z_correction=1):
        """Calculate spatial resolution for all datasets."""
        criterion = self.resolution_threshold
        threshold = self.threshold_value
        snr = self.snr_value
        # tolerance = self.resolution_point_sigma
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
            )

            # # # Find intersection
            # root, result = optimize.brentq(
            #     pdiff2 if criterion == 'fixed' else pdiff1,
            #     0.0, 1.0, xtol=tolerance, full_output=True)
            #
            # # Save result, if intersection was found
            # if result.converged is True:
            #     data_set.resolution["resolution-point"] = (frc_eq(root), root)
            #     data_set.resolution["criterion"] = criterion
            #     resolution = 2 * self.spacing / root
            #     data_set.resolution["resolution"] = resolution
            #     self.data_collection[int(key)] = data_set
            # else:
            #     print "Could not find an intersection for the curves for the dataset %s." % key

        return self.data_collection

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
    ):
        """Process a single dataset and return updated data."""
        if verbose:
            print(f"Calculating resolution point for dataset {key}")

        frc_eq = fit_frc_curve(data_set, degree, fit_type)
        two_sigma_eq = calculate_resolution_threshold_curve(
            data_set, criterion, threshold, snr
        )

        def pdiff1(x):
            return abs(frc_eq(x) - two_sigma_eq(x))

        def pdiff2(x):
            return abs(frc_eq(x) - threshold)

        def first_guess(x, y, thr):
            difference = y - thr
            indices = np.where(difference <= 0)[0]
            if len(indices) == 0:
                return np.nan
            return x[max(indices[0] - 1, 0)]

        fit_start = first_guess(
            data_set.correlation["frequency"],
            data_set.correlation["correlation"],
            np.mean(data_set.resolution["threshold"]),
        )

        if np.isnan(fit_start):
            data_set.resolution["resolution-point"] = (np.nan, np.nan)
            data_set.resolution["criterion"] = criterion
            data_set.resolution["resolution"] = np.nan
            data_set.resolution["spacing"] = self.spacing
        else:
            if verbose:
                print(f"Fit starts at {fit_start}")
                disp = 1
            else:
                disp = 0
            root = optimize.fmin(
                pdiff2 if criterion == "fixed" else pdiff1, fit_start, disp=disp
            )[0]

            data_set.resolution["resolution-point"] = (frc_eq(root), root)
            data_set.resolution["criterion"] = criterion

            angle = np.deg2rad(int(key))
            z_multiplier = 1 + (z_correction - 1) * np.abs(np.sin(angle))
            resolution = z_multiplier * (2 * self.spacing / root)

            data_set.resolution["resolution"] = resolution
            data_set.resolution["spacing"] = self.spacing * z_multiplier

        return data_set
