"""Sentinel test: package importable, version exposed."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_importable() -> None:
    import topicgpt

    assert hasattr(topicgpt, "__version__")


@pytest.mark.unit
def test_version_string_shape() -> None:
    from topicgpt import __version__

    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2
