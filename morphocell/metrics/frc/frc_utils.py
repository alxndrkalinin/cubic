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

"""Backward compatibility module for FRC utilities."""

from .analysis import (
    get_frc_options,
    FixedDictionary,
    safe_divide,
    FourierCorrelationDataCollection,
    FourierCorrelationData,
    fit_frc_curve,
    calculate_snr_threshold_value,
    calculate_resolution_threshold_curve,
    FourierCorrelationAnalysis,
)
from .iterators import (
    cast_to_dtype,
    rescale_to_min_max,
    expand_to_shape,
    FourierRingIterator,
    SectionedFourierRingIterator,
    FourierShellIterator,
    SectionedFourierShellIterator,
    HollowSectionedFourierShellIterator,
    AxialExcludeSectionedFourierShellIterator,
    RotatingFourierShellIterator,
)

__all__ = [
    "get_frc_options",
    "FixedDictionary",
    "safe_divide",
    "FourierCorrelationDataCollection",
    "FourierCorrelationData",
    "fit_frc_curve",
    "calculate_snr_threshold_value",
    "calculate_resolution_threshold_curve",
    "FourierCorrelationAnalysis",
    "cast_to_dtype",
    "rescale_to_min_max",
    "expand_to_shape",
    "FourierRingIterator",
    "SectionedFourierRingIterator",
    "FourierShellIterator",
    "SectionedFourierShellIterator",
    "HollowSectionedFourierShellIterator",
    "AxialExcludeSectionedFourierShellIterator",
    "RotatingFourierShellIterator",
]
