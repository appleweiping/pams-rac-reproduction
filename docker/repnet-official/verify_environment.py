"""Static/runtime checks for the isolated official RepNet environment."""

from __future__ import annotations

import os
import sys

import cv2
import keras
import numpy as np
import scipy
import tensorflow as tf


def main() -> None:
    assert sys.version_info[:3] == (3, 11, 10)
    assert tf.__version__ == "2.17.1"
    assert keras.__version__ == "3.5.0"
    assert np.__version__ == "1.26.4"
    assert cv2.__version__ == "4.10.0"
    assert scipy.__version__ == "1.14.1"

    build = tf.sysconfig.get_build_info()
    assert build["is_cuda_build"] is True
    assert build["cuda_version"] == "12.3"
    assert str(build["cudnn_version"]).startswith("8")

    # The official ec7c3d notebook's custom checkpoint mapping dereferences
    # `weight.value.shape.num_elements()` before assigning every old tensor.
    variable = tf.keras.Variable(
        tf.keras.initializers.TruncatedNormal(stddev=0.02),
        shape=(1, 64, 512),
        trainable=True,
        name="pos_encoding",
    )
    assert variable.value.shape.num_elements() == 64 * 512
    variable.assign(tf.zeros_like(variable))

    # Keras 3 must expose the named ResNet50V2 endpoint used by RepNet and
    # retain a deterministic ordered weight list for the notebook mapping.
    base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        pooling="max",
    )
    endpoint = base.get_layer("conv4_block3_out")
    assert endpoint.output.shape[-1] == 1024
    repnet_base = tf.keras.Model(inputs=base.input, outputs=endpoint.output)
    paths = tuple(weight.path for weight in repnet_base.weights)
    assert paths[:4] == (
        "conv1_conv/kernel",
        "conv1_conv/bias",
        "conv2_block1_preact_bn/gamma",
        "conv2_block1_preact_bn/beta",
    )
    assert paths[-2:] == (
        "conv4_block3_3_conv/kernel",
        "conv4_block3_3_conv/bias",
    )

    expected_policy = f"limit:{os.environ.get('REPNET_GPU_MEMORY_LIMIT_MB', '8192')}"
    assert os.environ.get("REPNET_TF_GPU_POLICY_APPLIED") == expected_policy
    for physical_gpu in tf.config.list_physical_devices("GPU"):
        logical = tf.config.get_logical_device_configuration(physical_gpu)
        assert logical is not None and len(logical) == 1
        assert logical[0].memory_limit == float(
            os.environ.get("REPNET_GPU_MEMORY_LIMIT_MB", "8192")
        )

    print(
        "repnet-official environment verified: "
        f"tensorflow={tf.__version__} keras={keras.__version__} "
        f"cuda={build['cuda_version']} cudnn={build['cudnn_version']} "
        f"gpu_policy={expected_policy}"
    )


if __name__ == "__main__":
    main()
