from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform

EXPECTED = {
    "einops": "0.3.2",
    "kornia": "0.5.11",
    "mmcv": "1.4.0",
    "timm": "0.4.12",
}


def main() -> None:
    torch = importlib.import_module("torch")
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        **{
            package: importlib.metadata.version(package)
            for package in EXPECTED
        },
    }
    for package, expected in EXPECTED.items():
        if versions[package] != expected:
            raise SystemExit(
                f"{package}={versions[package]} does not match {expected}"
            )
    if versions["torch"] != "2.5.1+cu124" or versions["cuda_runtime"] != "12.4":
        raise SystemExit("base Torch/CUDA runtime does not match the audited stack")
    if not bool(torch.cuda.is_available()):
        raise SystemExit("CUDA is required for official TransRAC inference")
    versions["gpu_name"] = str(torch.cuda.get_device_properties(0).name)
    print(json.dumps(versions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
