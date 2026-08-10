# Official RepNet ckpt-70 isolated environment

This image is only an environment for source-sanity execution of the official
RepNet notebook at Google Research revision
`ec7c3d346277b737bc2decffcd1b533d4b7ec105`. It contains no RepNet source,
checkpoint, UCFRep video, PAMS runner, or benchmark result.

At runtime the reproduction checkout is mounted at `/workspace`;
`PYTHONPATH=/opt/repnet-official:/workspace/src` makes the mounted `pams`
package importable without copying it into this independently locked image.
The runner can consume the hash-verified notebook file directly, so neither a
Git executable nor a Google Research checkout is required in the image.

The notebook does not pin TensorFlow. Its 30 June 2025 compatibility change
uses Keras 3 variables plus an explicit old-checkpoint tensor mapping.
TensorFlow `2.17.1`, Keras `3.5.0`, Python `3.11.10`, CUDA `12.3`, and cuDNN
`8.9.7` are an independently frozen compatibility choice. CUDA 12.3 is
driver-compatible with the target server's NVIDIA 550.54.14 driver, unlike a
CUDA 12.5 TensorFlow 2.18 stack.

The image defaults to one 8192 MiB TensorFlow logical-device limit. This is
applied from `sitecustomize.py` before user code imports TensorFlow and fails
closed if configuration cannot be applied. A deliberate override must remain
positive and bounded:

```bash
docker run --rm --gpus '"device=0"' \
  --env REPNET_GPU_MEMORY_LIMIT_MB=10240 \
  repnet-official:local
```

Build from the dedicated context:

```bash
docker build \
  --file docker/repnet-official/Dockerfile \
  --tag repnet-official:local \
  docker/repnet-official
```

Successful image construction verifies package versions, TensorFlow CUDA
metadata, the Keras variable API used by the custom mapping, the named
ResNet50V2 endpoint, and ordered Keras 3 weight paths. It does **not** prove
that ckpt-70 was completely restored. That claim requires a later offline
checkpoint-load smoke which checks every mapping entry, tensor shape, and
post-load forward output.

The runner executes only the notebook's selected model, checkpoint mapping,
OpenCV decoder, and counting definitions. Their non-standard imports are
NumPy, OpenCV, SciPy's median filter, and TensorFlow/Keras (including
TensorFlow's checkpoint reader), all of which are hash-locked here. Notebook
front-end packages such as Jupyter, IPython, `nbformat`, and Matplotlib are
not executed. Video input goes through `cv2.VideoCapture`; the runner never
invokes an external `ffmpeg` program. Consequently the image intentionally
adds no operating-system packages or Git client.
