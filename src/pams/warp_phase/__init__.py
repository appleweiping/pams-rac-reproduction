"""Isolated WARP-PHASE prospective-pilot package.

The initializer is intentionally dependency-free.  In particular, importing a
numerical submodule such as :mod:`pams.warp_phase.selector` in the dedicated
fixture environment must not import PyYAML, Pydantic, the data reader, or the
trusted offline packer as a side effect.
"""

PACKAGE_SCHEMA_VERSION = 1

__all__ = ["PACKAGE_SCHEMA_VERSION"]
