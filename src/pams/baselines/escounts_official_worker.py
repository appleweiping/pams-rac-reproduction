"""Python 3.8-compatible JSONL worker for the frozen ESCounts runtime.

This file deliberately does not import :mod:`pams`.  The audited host process
uses Python 3.10 or newer, while the released ESCounts dependency stack is
frozen to Python 3.8 and PyTorch 1.10.  Communication is one request and one
response per line on stdin/stdout; model and dependency diagnostics are sent
to stderr so they cannot corrupt the protocol stream.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import stat
import sys
from pathlib import Path

PROTOCOL_VERSION = 1
OFFICIAL_SOURCE_COMMIT = "f18fcf1933abb3d4f199fd12ee3cafc53581576a"
OFFICIAL_SOURCE_TREE = "3bedc8d117c4c86954fb58d891326e7c9c6a1a3d"
PYTORCHVIDEO_COMMIT = "fae0d89a194a2c1ca99e59eab6eedd40bde38726"
PYTORCHVIDEO_TREE = "bd12d614191c96f3625f111284dec2b0a56a178a"
ENCODER_SHA256 = "b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8"
DECODER_SHA256 = "297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559"
CONFIG_SHA256 = "f16afd40c8661761817bad7d2bed7504da29f27f0084034745d5a82822bf2974"
ENCODER_BYTES = 1_207_498_009
DECODER_BYTES = 239_106_669
PRIMARY_MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
RETRY_MEMORY_LIMIT_BYTES = 12 * 1024 * 1024 * 1024
ALLOWED_MEMORY_LIMITS = {
    PRIMARY_MEMORY_LIMIT_BYTES,
    RETRY_MEMORY_LIMIT_BYTES,
}
OFFICIAL_CRITICAL_FILES = {
    "demo.py": (7_368, "bce255b1274d6f3feedd42cf671125e844354f158c6865590e302fd704216f38"),
    "video_mae_cross_full_attention.py": (
        21_511,
        "e1597a0ec2fcf5f76deb7b1396a9059de8b0edcca7ff8a64ba25762b5f498ace",
    ),
    "configs/pretrain_config.yaml": (
        1_876,
        "38960518be9cd640573c58a5aaf3e850f721dc92d445d5b8e787ddbbfed0bf67",
    ),
    "slowfast/utils/parser.py": (
        3_051,
        "76b08fd4392ccbe103c76480517cd2d1219033339756a7f47c71a78b4113d218",
    ),
}
SHA256_CHARACTERS = frozenset("0123456789abcdef")


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_object(value, expected, field):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{field} fields are not frozen")
    return value


def validate_sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def stable_file_digest(path):
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, closed, after)
    }
    if len(identities) != 1 or byte_count != before.st_size:
        raise RuntimeError(f"file changed while hashed: {path}")
    return {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "identity": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ),
    }


def assert_digest_unchanged(path, expected, role):
    observed = stable_file_digest(path)
    if observed != expected:
        raise RuntimeError(f"{role} changed during worker lifetime")


def relative_python_hashes(root):
    output = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        output[relative] = stable_file_digest(path)["sha256"]
    if not output:
        raise RuntimeError(f"Python source tree is empty: {root}")
    return output


def verify_pytorchvideo_origin(checkout_root, imported_package):
    module_file = Path(imported_package.__file__).resolve(strict=True)
    installed_root = module_file.parent
    checkout_package = checkout_root / "pytorchvideo"
    checkout_hashes = relative_python_hashes(checkout_package)
    installed_hashes = relative_python_hashes(installed_root)
    required = {"__init__.py", "data/utils.py", "transforms/__init__.py"}
    if (
        not required.issubset(installed_hashes)
        or not set(installed_hashes).issubset(checkout_hashes)
        or any(
            installed_hashes[path] != checkout_hashes[path]
            for path in installed_hashes
        )
    ):
        raise RuntimeError(
            "installed PyTorchVideo Python sources differ from frozen checkout"
        )
    return {
        "repository_commit": PYTORCHVIDEO_COMMIT,
        "repository_tree": PYTORCHVIDEO_TREE,
        "checkout_root": str(checkout_root),
        "installed_root": str(installed_root),
        "python_file_count": len(installed_hashes),
        "python_tree_sha256": canonical_sha256(installed_hashes),
        "checkout_python_file_count": len(checkout_hashes),
    }


@contextlib.contextmanager
def source_import_context(root):
    previous_cwd = Path.cwd()
    inserted = str(root)
    sys.path.insert(0, inserted)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(str(previous_cwd))
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)


def ensure_module_under(module, root, role):
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"{role} module has no source path")
    resolved = Path(module_file).resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise RuntimeError(f"{role} module is outside frozen source: {resolved}")
    return {
        "path": str(resolved),
        "sha256": stable_file_digest(resolved)["sha256"],
    }


class DecodeError(RuntimeError):
    pass


class ResourceExhaustedError(RuntimeError):
    pass


class OfficialRuntime:
    def __init__(self, arguments):
        self.source_root = arguments.source_root.resolve(strict=True)
        self.pytorchvideo_source_root = arguments.pytorchvideo_source_root.resolve(
            strict=True
        )
        self.video_root = arguments.video_root.resolve(strict=True)
        self.encoder_path = arguments.encoder.resolve(strict=True)
        self.decoder_path = arguments.decoder.resolve(strict=True)
        self.device = arguments.device
        self.memory_limit_bytes = arguments.memory_limit_bytes
        self.container_image_id = arguments.container_image_id
        if self.memory_limit_bytes not in ALLOWED_MEMORY_LIMITS:
            raise ValueError("memory limit is not an authorized ESCounts tier")
        if (
            not self.container_image_id.startswith("sha256:")
            or len(self.container_image_id) != 71
        ):
            raise ValueError("container image ID must be a full sha256:<digest>")
        if os.environ.get("ESCOUNTS_IMAGE_ID") != self.container_image_id:
            raise RuntimeError("worker container image environment binding mismatch")

        self.source_identity = {}
        for relative, expected in OFFICIAL_CRITICAL_FILES.items():
            digest = stable_file_digest(self.source_root / relative)
            if (digest["bytes"], digest["sha256"]) != expected:
                raise RuntimeError(
                    f"official critical source identity mismatch: {relative}"
                )
            self.source_identity[relative] = digest
        self.encoder_identity = stable_file_digest(self.encoder_path)
        self.decoder_identity = stable_file_digest(self.decoder_path)
        if (
            self.encoder_identity["sha256"] != ENCODER_SHA256
            or self.encoder_identity["bytes"] != ENCODER_BYTES
            or self.decoder_identity["sha256"] != DECODER_SHA256
            or self.decoder_identity["bytes"] != DECODER_BYTES
        ):
            raise RuntimeError("official checkpoint identity mismatch")

        dependencies = {}
        for name in ("av", "cv2", "einops", "numpy", "torch", "torchvision"):
            dependencies[name] = importlib.import_module(name)
        pytorchvideo_package = importlib.import_module("pytorchvideo")
        pytorchvideo_data = importlib.import_module("pytorchvideo.data.utils")
        pytorchvideo_transforms = importlib.import_module("pytorchvideo.transforms")
        self.pytorchvideo_origin = verify_pytorchvideo_origin(
            self.pytorchvideo_source_root,
            pytorchvideo_package,
        )
        with source_import_context(self.source_root):
            official_model = importlib.import_module(
                "video_mae_cross_full_attention"
            )
            parser_module = importlib.import_module("slowfast.utils.parser")
        self.module_origins = {
            "official_model": ensure_module_under(
                official_model,
                self.source_root,
                "official model",
            ),
            "slowfast_parser": ensure_module_under(
                parser_module,
                self.source_root,
                "SlowFast parser",
            ),
        }
        slowfast_files = {}
        for name, module in sorted(sys.modules.items()):
            if name != "slowfast" and not name.startswith("slowfast."):
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            origin = ensure_module_under(
                module,
                self.source_root,
                f"loaded module {name}",
            )
            slowfast_files[name] = origin["sha256"]
        if not slowfast_files:
            raise RuntimeError("no frozen SlowFast modules were imported")
        self.module_origins["slowfast_loaded_modules"] = {
            "module_count": len(slowfast_files),
            "sha256": canonical_sha256(slowfast_files),
        }

        self.av = dependencies["av"]
        self.cv2 = dependencies["cv2"]
        self.einops = dependencies["einops"]
        self.np = dependencies["numpy"]
        self.torch = dependencies["torch"]
        self.thwc_to_cthw = pytorchvideo_data.thwc_to_cthw
        self.cuda = self.device.startswith("cuda")
        if self.cuda:
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            self.torch.cuda.set_device(index)
            total = self.torch.cuda.get_device_properties(index).total_memory
            self.torch.cuda.set_per_process_memory_fraction(
                float(self.memory_limit_bytes) / float(total),
                index,
            )
        elif self.device != "cpu":
            raise ValueError("device must be cpu or cuda:<index>")

        class Args:
            opts = None

        with source_import_context(self.source_root):
            cfg = parser_module.load_config(
                Args(),
                path_to_config=str(
                    self.source_root / "configs/pretrain_config.yaml"
                ),
            )
        model_class = official_model.SupervisedMAE
        encoder = model_class(
            cfg=cfg,
            just_encode=True,
            use_precomputed=False,
            encodings="mae",
        ).to(self.device)
        encoder_checkpoint = self.torch.load(
            str(self.encoder_path),
            map_location=self.device,
        )
        encoder_state = encoder_checkpoint["model_state"]
        unmatched = []
        with self.torch.no_grad():
            for name, target in encoder.state_dict().items():
                if "decode" in name:
                    continue
                if name in encoder_state:
                    target.copy_(encoder_state[name])
                    continue
                if ".qkv." in name and "blocks" in name:
                    q_name = name.replace(".qkv.", ".q.").replace("module.", "")
                    k_name = name.replace(".qkv.", ".k.").replace("module.", "")
                    v_name = name.replace(".qkv.", ".v.").replace("module.", "")
                    if all(
                        key in encoder_state for key in (q_name, k_name, v_name)
                    ):
                        target.copy_(
                            self.torch.cat(
                                [
                                    encoder_state[q_name],
                                    encoder_state[k_name],
                                    encoder_state[v_name],
                                ]
                            )
                        )
                        continue
                unmatched.append(name)
        if unmatched:
            raise RuntimeError(
                f"encoder checkpoint left {len(unmatched)} tensors unmatched"
            )
        encoder.eval()

        decoder = model_class(
            cfg=cfg,
            use_precomputed=True,
            token_pool_ratio=0.4,
            iterative_shots=True,
            encodings="mae",
            no_exemplars=False,
            window_size=(4, 7, 7),
        ).to(self.device)
        decoder_checkpoint = self.torch.load(
            str(self.decoder_path),
            map_location=self.device,
        )
        decoder.load_state_dict(
            decoder_checkpoint["model_state_dict"],
            strict=True,
        )
        decoder.eval()
        self.encoder = encoder
        self.decoder = decoder
        self.transform = pytorchvideo_transforms.create_video_transform(
            mode="test",
            convert_to_float=False,
            min_size=224,
            crop_size=224,
            num_samples=None,
            video_mean=[0.485, 0.456, 0.406],
            video_std=[0.229, 0.224, 0.225],
        )
        self.runtime_versions = {
            "python": sys.version.split()[0],
            "torch": str(self.torch.__version__),
            "torchvision": str(dependencies["torchvision"].__version__),
            "cuda": str(self.torch.version.cuda),
            "numpy": str(dependencies["numpy"].__version__),
            "opencv": str(dependencies["cv2"].__version__),
            "av": str(dependencies["av"].__version__),
            "container_image_id": self.container_image_id,
            "pytorchvideo_commit": PYTORCHVIDEO_COMMIT,
        }

    def assert_critical_inputs_unchanged(self):
        for relative, digest in self.source_identity.items():
            assert_digest_unchanged(
                self.source_root / relative,
                digest,
                f"official source {relative}",
            )
        assert_digest_unchanged(
            self.encoder_path,
            self.encoder_identity,
            "official encoder",
        )
        assert_digest_unchanged(
            self.decoder_path,
            self.decoder_identity,
            "official decoder",
        )
        observed_pytorchvideo_origin = verify_pytorchvideo_origin(
            self.pytorchvideo_source_root,
            importlib.import_module("pytorchvideo"),
        )
        if observed_pytorchvideo_origin != self.pytorchvideo_origin:
            raise RuntimeError("PyTorchVideo sources changed during worker lifetime")

    def resolve_video(self, locator):
        if (
            not isinstance(locator, str)
            or not locator
            or "\\" in locator
            or locator.startswith("/")
            or any(part in ("", ".", "..") for part in locator.split("/"))
        ):
            raise ValueError("video locator is not a safe portable relative path")
        path = (self.video_root / Path(locator)).resolve(strict=True)
        if self.video_root != path and self.video_root not in path.parents:
            raise ValueError("video locator escapes worker video root")
        return path

    def read_video(self, path):
        capture = self.cv2.VideoCapture(str(path))
        try:
            frame_count = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if frame_count < 1:
            raise DecodeError("OpenCV reported no video frames")
        frames = []
        try:
            container = self.av.open(str(path))
            try:
                for index, frame in enumerate(container.decode(video=0)):
                    if index >= frame_count:
                        break
                    frames.append(frame)
            finally:
                container.close()
        except Exception as error:
            raise DecodeError(f"PyAV decode failed: {error}") from error
        if len(frames) != frame_count:
            raise DecodeError(
                f"decoded {len(frames)} of {frame_count} expected frames"
            )
        try:
            tensors = [
                self.torch.from_numpy(frame.to_ndarray(format="rgb24"))
                for frame in frames
            ]
            video = self.thwc_to_cthw(
                self.torch.stack(tensors).to(self.torch.float32)
            )
        except Exception as error:
            raise DecodeError(f"RGB tensor conversion failed: {error}") from error
        return video, len(frames)

    def predict(self, path):
        try:
            video, decoded_frames = self.read_video(path)
            frames = self.transform(video / 255.0)
            channels, total_frames, height, width = frames.shape
            padding = self.torch.zeros([channels, 64, height, width])
            frames = self.torch.cat([frames, padding], dim=1)
            clips = []
            for start in range(0, total_frames, 16):
                indices = self.np.linspace(
                    start,
                    start + 64,
                    17,
                )[:16].astype(int)
                clips.append(frames[:, indices])
            data = self.torch.stack(clips).to(self.device)
            device_type = "cuda" if self.cuda else "cpu"
            with self.torch.autocast(
                enabled=self.cuda,
                device_type=device_type,
            ), self.torch.no_grad():
                encoded, thw = self.encoder(data)
                encoded = encoded.transpose(1, 2).reshape(
                    encoded.shape[0],
                    encoded.shape[-1],
                    thw[0],
                    thw[1],
                    thw[2],
                )
            encoded = encoded[0::4]
            encoded = self.einops.rearrange(
                encoded,
                "S C T H W -> C (S T) H W",
            )
            factor = math.ceil(encoded.shape[-1] * 0.4)
            tokens = self.torch.nn.functional.adaptive_avg_pool3d(
                encoded,
                (encoded.shape[-3], factor, factor),
            )
            tokens = tokens.unsqueeze(0)
            shapes = tokens.shape[-3:]
            tokens = self.einops.rearrange(
                tokens,
                "B C T H W -> B (T H W) C",
            )
            with self.torch.autocast(
                enabled=self.cuda,
                device_type=device_type,
            ), self.torch.no_grad():
                density = self.decoder(tokens, thw=[shapes], shot_num=0)
            raw_count = float(density.sum().item() / 100.0)
        except DecodeError:
            raise
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if self.cuda:
                    self.torch.cuda.empty_cache()
                raise ResourceExhaustedError(str(error)) from error
            raise
        finally:
            if self.cuda:
                self.torch.cuda.empty_cache()
        if not math.isfinite(raw_count) or raw_count < 0.0:
            raise RuntimeError("official decoder returned invalid count")
        return raw_count, decoded_frames


def load_message(line, expected_type):
    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSONL request field: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSONL request constant: {value}")
            ),
        )
    except Exception as error:
        raise ValueError(f"invalid JSONL request: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    supplied = payload.pop("message_sha256", None)
    validate_sha256(supplied, "message_sha256")
    if supplied != canonical_sha256(payload):
        raise ValueError("request message SHA-256 mismatch")
    if payload.get("schema_version") != PROTOCOL_VERSION or (
        expected_type is not None and payload.get("type") != expected_type
    ):
        raise ValueError("unexpected JSONL request type")
    return payload


def send_message(payload):
    output = dict(payload)
    output["message_sha256"] = canonical_sha256(output)
    sys.stdout.write(
        json.dumps(
            output,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    sys.stdout.flush()


def worker_main(arguments):
    sys.dont_write_bytecode = True
    worker_path = Path(__file__).resolve(strict=True)
    worker_digest = stable_file_digest(worker_path)
    init_line = sys.stdin.readline()
    if not init_line:
        raise RuntimeError("worker stdin closed before initialization")
    init = load_message(init_line, "init")
    strict_object(
        init,
        {
            "schema_version",
            "type",
            "nonce",
            "source_commit",
            "source_tree",
            "pytorchvideo_commit",
            "pytorchvideo_tree",
            "encoder_sha256",
            "decoder_sha256",
            "config_sha256",
            "worker_sha256",
            "container_image_id",
            "memory_limit_bytes",
            "device",
        },
        "initialization request",
    )
    if (
        init["source_commit"] != OFFICIAL_SOURCE_COMMIT
        or init["source_tree"] != OFFICIAL_SOURCE_TREE
        or init["pytorchvideo_commit"] != PYTORCHVIDEO_COMMIT
        or init["pytorchvideo_tree"] != PYTORCHVIDEO_TREE
        or init["encoder_sha256"] != ENCODER_SHA256
        or init["decoder_sha256"] != DECODER_SHA256
        or init["config_sha256"] != CONFIG_SHA256
        or init["worker_sha256"] != worker_digest["sha256"]
        or init["container_image_id"] != arguments.container_image_id
        or init["memory_limit_bytes"] != arguments.memory_limit_bytes
        or init["device"] != arguments.device
    ):
        raise RuntimeError("initialization binding mismatch")

    with contextlib.redirect_stdout(sys.stderr):
        runtime = OfficialRuntime(arguments)
    send_message(
        {
            "schema_version": PROTOCOL_VERSION,
            "type": "ready",
            "nonce": init["nonce"],
            "worker_sha256": worker_digest["sha256"],
            "container_image_id": runtime.container_image_id,
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "source_tree": OFFICIAL_SOURCE_TREE,
            "pytorchvideo_origin": runtime.pytorchvideo_origin,
            "module_origins": runtime.module_origins,
            "runtime_versions": runtime.runtime_versions,
        }
    )

    for line in sys.stdin:
        request = load_message(line, None)
        request_type = request.get("type")
        if request_type == "shutdown":
            strict_object(
                request,
                {"schema_version", "type", "nonce"},
                "shutdown request",
            )
            runtime.assert_critical_inputs_unchanged()
            assert_digest_unchanged(
                worker_path,
                worker_digest,
                "worker code",
            )
            send_message(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "type": "shutdown_ack",
                    "nonce": request["nonce"],
                    "worker_sha256": worker_digest["sha256"],
                }
            )
            return 0
        strict_object(
            request,
            {
                "schema_version",
                "type",
                "request_id",
                "video_id",
                "video_locator",
                "video_sha256",
            },
            "prediction request",
        )
        if request_type != "predict":
            raise ValueError("unexpected worker request")
        validate_sha256(request["video_sha256"], "video_sha256")
        path = runtime.resolve_video(request["video_locator"])
        before = stable_file_digest(path)
        if before["sha256"] != request["video_sha256"]:
            raise RuntimeError("worker video SHA-256 differs from host request")
        response = {
            "schema_version": PROTOCOL_VERSION,
            "type": "result",
            "request_id": request["request_id"],
            "video_id": request["video_id"],
            "video_sha256": request["video_sha256"],
        }
        try:
            with contextlib.redirect_stdout(sys.stderr):
                raw_count, decoded_frames = runtime.predict(path)
            response.update(
                {
                    "status": "ok",
                    "raw_count": raw_count,
                    "decoded_frames": decoded_frames,
                    "error_type": None,
                    "error_message": None,
                }
            )
        except ResourceExhaustedError as error:
            response.update(
                {
                    "status": "resource_exhausted",
                    "raw_count": None,
                    "decoded_frames": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        except DecodeError as error:
            response.update(
                {
                    "status": "decode_failed",
                    "raw_count": None,
                    "decoded_frames": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        except Exception as error:
            response.update(
                {
                    "status": "inference_failed",
                    "raw_count": None,
                    "decoded_frames": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        after = stable_file_digest(path)
        if before != after:
            raise RuntimeError("worker video changed during inference")
        send_message(response)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pytorchvideo-source-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--container-image-id", required=True)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        return worker_main(arguments)
    except Exception as error:
        print(
            f"fatal ESCounts worker error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
