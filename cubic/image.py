"""Contains a simple class for storing image data."""

from typing import Any
from collections.abc import Sequence

from .cuda import to_device, get_device
from .skimage import util


class Image:
    """A simple class to store NumPy/CuPy image data and metadata."""

    def __init__(
        self,
        images: Any,
        spacing: Sequence[float | int],
        filename: str | None = None,
        device: str | None = None,
        as_float: bool = True,
    ) -> None:
        """Create image object with data stored either as NumPy array (CPU) or CuPy array (GPU)."""
        if device is None:
            device = get_device(images)

        # ``to_device`` raises ValueError for anything other than "CPU"/"GPU".
        self.data = to_device(images, device)

        if as_float:
            self.data = util.img_as_float32(self.data)

        self.shape = self.data.shape
        self.spacing = tuple(spacing)
        self.filename = filename
        self.device = device

    def to_gpu(self) -> None:
        """Move Image data to GPU."""
        self.data = to_device(self.data, "GPU")
        self.device = "GPU"

    def to_cpu(self) -> None:
        """Move (or keep) Image on CPU."""
        self.data = to_device(self.data, "CPU")
        self.device = "CPU"
