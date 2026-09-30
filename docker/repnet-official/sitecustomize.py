"""Fail-closed TensorFlow GPU policy for the isolated RepNet environment."""

from __future__ import annotations

import os
import sys


def _fail(message: str) -> None:
    sys.stderr.write(f"repnet GPU policy failure: {message}\n")
    sys.stderr.flush()
    os._exit(78)


def _configured_limit_mb() -> float:
    raw = os.environ.get("REPNET_GPU_MEMORY_LIMIT_MB", "8192").strip()
    try:
        limit = float(raw)
    except ValueError:
        _fail("REPNET_GPU_MEMORY_LIMIT_MB must be a finite positive number")
    if not 512.0 <= limit <= 49140.0:
        _fail("REPNET_GPU_MEMORY_LIMIT_MB must be between 512 and 49140 MiB")
    return limit


try:
    import tensorflow as tf

    memory_limit_mb = _configured_limit_mb()
    physical_gpus = tf.config.list_physical_devices("GPU")
    for physical_gpu in physical_gpus:
        tf.config.set_logical_device_configuration(
            physical_gpu,
            [
                tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit_mb,
                )
            ],
        )
    os.environ["REPNET_TF_GPU_POLICY_APPLIED"] = f"limit:{memory_limit_mb:g}"
except BaseException as exc:  # pragma: no cover - exercised inside the image
    _fail(f"{type(exc).__name__}: {exc}")
