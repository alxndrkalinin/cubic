"""Tests for the Cellpose segmentation wrapper."""

import sys
import types
import builtins
import warnings
import importlib
from typing import Any

import numpy as np
import pytest

from cubic.segmentation import cellpose as cellpose_mod
from cubic.segmentation.cellpose import cellpose_eval, cellpose_segment


class _FakeModel:
    """Stand-in for ``cellpose.models.CellposeModel`` recording construction."""

    def __init__(self, gpu: bool = True, pretrained_model: str = "cpsam") -> None:
        self.gpu = gpu
        self.pretrained_model = pretrained_model

    def eval(self, image: np.ndarray, **kwargs: Any) -> tuple[Any, None, None]:
        return np.zeros(image.shape, dtype=np.int32), None, None


@pytest.fixture
def fake_cellpose(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, str]]:
    """Install a fake ``models`` module and return the construction log."""
    built: list[tuple[bool, str]] = []

    class Recorded(_FakeModel):
        def __init__(self, gpu: bool = True, pretrained_model: str = "cpsam") -> None:
            super().__init__(gpu=gpu, pretrained_model=pretrained_model)
            built.append((gpu, pretrained_model))

    # raising=False so the fixture also works where cellpose is not installed
    # and the module-level ``models`` import never bound a name
    monkeypatch.setattr(
        cellpose_mod,
        "models",
        types.SimpleNamespace(CellposeModel=Recorded),
        raising=False,
    )
    monkeypatch.setattr(cellpose_mod, "_CELLPOSE_AVAILABLE", True)
    monkeypatch.setattr(cellpose_mod, "_MODEL_CACHE", {})
    return built


def test_cellpose_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cellpose should raise if the library is unavailable.

    Force the unavailable branch so the test is deterministic regardless of
    whether cellpose is installed in the running environment.
    """
    monkeypatch.setattr(cellpose_mod, "_CELLPOSE_AVAILABLE", False)
    image = np.zeros((1, 32, 32), dtype=np.float32)
    with pytest.raises(ImportError):
        cellpose_segment(image)
    with pytest.raises(ImportError):
        cellpose_eval(image)


def test_import_without_cellpose_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the module without cellpose is silent; it raises at call time.

    The module used to ``warnings.warn`` at import, firing for every user of
    ``import cubic.segmentation`` who has no cellpose installed.
    """
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cellpose" or name.startswith("cellpose."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.delitem(sys.modules, "cubic.segmentation.cellpose")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = importlib.import_module("cubic.segmentation.cellpose")

    assert reloaded._CELLPOSE_AVAILABLE is False


def test_cellpose_eval_caches_model(fake_cellpose: list[tuple[bool, str]]) -> None:
    """The model is built once per (pretrained_model, gpu) pair, not per call."""
    image = np.zeros((2, 16, 16), dtype=np.float32)

    cellpose_eval(image)
    cellpose_eval(image)
    assert fake_cellpose == [(True, "cpsam")]

    cellpose_eval(image, gpu=False)
    assert fake_cellpose == [(True, "cpsam"), (False, "cpsam")]

    cellpose_eval(image, pretrained_model="cpdino")
    assert len(fake_cellpose) == 3


def test_cellpose_eval_uses_prebuilt_model(
    fake_cellpose: list[tuple[bool, str]],
) -> None:
    """A pre-built model is run as-is, without constructing another one."""
    image = np.zeros((2, 16, 16), dtype=np.float32)
    masks = cellpose_eval(image, model=_FakeModel())
    assert fake_cellpose == []
    assert masks.shape == image.shape


def test_cellpose_eval_prebuilt_model_skips_availability_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied model works even when the import guard says otherwise."""
    monkeypatch.setattr(cellpose_mod, "_CELLPOSE_AVAILABLE", False)
    image = np.zeros((2, 16, 16), dtype=np.float32)
    assert cellpose_eval(image, model=_FakeModel()).shape == image.shape


def test_cellpose_segment_rejects_border_value() -> None:
    """The inert ``border_value`` parameter is gone from ``cellpose_segment``."""
    image = np.zeros((2, 16, 16), dtype=np.float32)
    with pytest.raises(TypeError):
        cellpose_segment(image, border_value=100)  # type: ignore[call-arg]


def test_cellpose_segment_returns_downscaled_masks(
    fake_cellpose: list[tuple[bool, str]],
) -> None:
    """Masks come back on the ``downscale_factor`` grid, as documented."""
    image = np.zeros((4, 32, 32), dtype=np.float32)
    masks = cellpose_segment(image, downscale_factor=0.5, min_size=1)
    assert masks.shape == (4, 16, 16)
