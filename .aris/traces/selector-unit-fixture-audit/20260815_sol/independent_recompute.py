"""Independent read-only audit of the Amendment-001 selector unit-fixture pack."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


ROOT = Path(__file__).resolve().parents[4]
PACK = ROOT / "data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1"
SPECS = {
    "offbin_delta1_p20": (256, 20, 255, 127, 20),
    "lower_endpoint_p4": (64, 4, 63, 31, 4),
    "upper_endpoint_span126_p63": (127, 63, 126, 63, 63),
}
SUFFIX_SCHEMAS = {
    "input_clocks.npy": ("<i8", lambda n: (n,)),
    "input_pose.npy": ("<f8", lambda n: (n, 17, 3)),
    "input_joint_mask.npy": ("|u1", lambda n: (n, 17)),
    "expected_scale.npy": ("<f8", lambda n: (1,)),
    "expected_normalized_coordinates.npy": ("<f8", lambda n: (n, 17, 2)),
    "expected_normalized_joint_mask.npy": ("|u1", lambda n: (n, 17)),
    "expected_feature_frame_valid.npy": ("|u1", lambda n: (n,)),
    "expected_velocity.npy": ("<f8", lambda n: (n - 1, 34)),
    "expected_velocity_mask.npy": ("|u1", lambda n: (n - 1, 34)),
    "expected_fft_x.npy": ("<f8", lambda n: (256,)),
    "expected_fft_mask.npy": ("|u1", lambda n: (256, 34)),
    "expected_fft_power.npy": ("<f8", lambda n: (127,)),
    "expected_candidate_present.npy": ("|u1", lambda n: (125,)),
    "expected_acf.npy": ("<f8", lambda n: (125,)),
    "expected_acf_computed.npy": ("|u1", lambda n: (125,)),
    "expected_harmonic_power.npy": ("<f8", lambda n: (125, 3)),
    "expected_score.npy": ("<f8", lambda n: (125,)),
    "expected_score_computed.npy": ("|u1", lambda n: (125,)),
    "expected_score_valid.npy": ("|u1", lambda n: (125,)),
    "expected_local_max.npy": ("|u1", lambda n: (125,)),
    "expected_selected_period.npy": ("<i2", lambda n: (1,)),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def constructor(clock_count: int, source_period: int) -> tuple[np.ndarray, ...]:
    clocks = np.arange(clock_count, dtype="<i8")
    pose = np.zeros((clock_count, 17, 3), dtype="<f8")
    mask = np.ones((clock_count, 17), dtype="|u1")
    pose[:, :, 2] = np.float64(0.9)
    pose[:, 11, :2] = (-0.5, 0.0)
    pose[:, 12, :2] = (0.5, 0.0)
    pose[:, 5, :2] = (-0.5, 2.0)
    pose[:, 6, :2] = (0.5, 2.0)
    for clock_index in range(clock_count):
        for joint in range(17):
            if joint in {5, 6, 11, 12}:
                continue
            base_x = np.float64(joint % 5 - 2) / np.float64(4.0)
            base_y = np.float64(joint // 5) / np.float64(4.0)
            theta = (
                np.float64(2.0)
                * np.pi
                * np.float64(clocks[clock_index])
                / np.float64(source_period)
                + np.float64(joint) * np.float64(0.1)
            )
            pose[clock_index, joint, 0] = base_x + np.float64(0.2) * np.sin(theta)
            pose[clock_index, joint, 1] = base_y + np.float64(0.2) * np.cos(theta)
    return clocks, pose, mask


def normalize(pose: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray, ...]:
    valid = mask.astype(bool) & np.isfinite(pose).all(axis=2) & (pose[:, :, 2] >= 0.2)
    roots = np.zeros((pose.shape[0], 2), dtype=np.float64)
    root_valid = np.zeros(pose.shape[0], dtype=bool)
    scale_samples: list[float] = []
    for index in range(pose.shape[0]):
        hips = [joint for joint in (11, 12) if valid[index, joint]]
        if not hips:
            continue
        root = np.mean(pose[index, hips, :2], axis=0, dtype=np.float64)
        roots[index] = root
        root_valid[index] = True
        shoulders = [joint for joint in (5, 6) if valid[index, joint]]
        if shoulders:
            shoulder = np.mean(pose[index, shoulders, :2], axis=0, dtype=np.float64)
            distance = float(np.linalg.norm(shoulder - root))
            if math.isfinite(distance) and distance > 0.0:
                scale_samples.append(distance)
    scale = max(float(np.median(np.asarray(scale_samples, dtype=np.float64))), 1e-3)
    valid &= root_valid[:, None]
    coordinates = np.zeros((pose.shape[0], 17, 2), dtype=np.float64)
    centered = (pose[:, :, :2] - roots[:, None, :]) / scale
    coordinates[valid] = centered[valid]
    return scale, coordinates, valid, valid.sum(axis=1) >= 8


def query_velocity(
    clocks: np.ndarray,
    velocity: np.ndarray,
    velocity_mask: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((query.size, 34), dtype=np.float64)
    valid = np.zeros((query.size, 34), dtype=bool)
    inside = np.isfinite(query) & (query >= clocks[0]) & (query <= clocks[-1])
    cells = np.searchsorted(clocks, query[inside], side="right") - 1
    cells = np.minimum(cells, velocity.shape[0] - 1)
    cells = np.maximum(cells, 0)
    values[inside] = velocity[cells]
    valid[inside] = velocity_mask[cells]
    values[~valid] = 0.0
    return values, valid


def interpolate_power(power: np.ndarray, continuous_bin: float) -> float:
    if not math.isfinite(continuous_bin) or continuous_bin <= 0.0 or continuous_bin >= 128.0:
        return 0.0
    lower = math.floor(continuous_bin)
    upper = math.ceil(continuous_bin)
    if lower == upper:
        return float(power[lower])
    return float(
        (upper - continuous_bin) * power[lower]
        + (continuous_bin - lower) * power[upper]
    )


def recompute_selector(clocks: np.ndarray, coordinates: np.ndarray, joint_valid: np.ndarray) -> dict[str, np.ndarray]:
    q = clocks.astype(np.float64)
    flat = coordinates.reshape(q.size, 34)
    coordinate_valid = np.repeat(joint_valid, 2, axis=1)
    velocity_mask = coordinate_valid[:-1] & coordinate_valid[1:]
    raw_velocity = np.diff(flat, axis=0) / np.diff(q)[:, None]
    velocity_mask &= np.isfinite(raw_velocity)
    velocity = np.zeros_like(raw_velocity)
    velocity[velocity_mask] = raw_velocity[velocity_mask]
    span = float(q[-1] - q[0])
    p_max = min(128, math.floor(span / 2.0))
    candidates = np.arange(4, p_max + 1, dtype=np.int64)
    fft_x = q[0] + np.arange(256, dtype=np.float64) * (span / 255.0)
    fft_velocity, fft_mask = query_velocity(q, velocity, velocity_mask, fft_x)
    counts = fft_mask.sum(axis=0, dtype=np.int64)
    means = np.zeros(34, dtype=np.float64)
    nonempty = counts > 0
    means[nonempty] = fft_velocity[:, nonempty].sum(axis=0, dtype=np.float64) / counts[nonempty]
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(256, dtype=np.float64) / 256.0)
    weighted = fft_mask * hann[:, None] * (fft_velocity - means[None, :])
    weighted[~fft_mask] = 0.0
    denominator = float(np.sum(fft_mask * hann[:, None] ** 2, dtype=np.float64))
    spectrum = np.fft.rfft(weighted, n=256, axis=0, norm="forward")
    power = np.sum(np.abs(spectrum) ** 2, axis=1, dtype=np.float64) / denominator
    spectral_denominator = float(np.sum(power[1:128], dtype=np.float64))
    acf = np.full(candidates.shape, np.nan, dtype=np.float64)
    score = np.full(candidates.shape, np.nan, dtype=np.float64)
    score_valid = np.zeros(candidates.shape, dtype=bool)
    harmonic = np.zeros((candidates.size, 3), dtype=np.float64)
    spacing = span / 255.0
    for index, period_value in enumerate(candidates):
        period = float(period_value)
        support = fft_x + period <= q[-1]
        left, left_valid = query_velocity(q, velocity, velocity_mask, fft_x[support])
        right, right_valid = query_velocity(q, velocity, velocity_mask, fft_x[support] + period)
        pair_valid = left_valid & right_valid
        if np.any(pair_valid, axis=1).sum() < 16:
            continue
        numerator = float(np.sum(pair_valid * left * right, dtype=np.float64))
        left_energy = float(np.sum(pair_valid * left**2, dtype=np.float64))
        right_energy = float(np.sum(pair_valid * right**2, dtype=np.float64))
        correlation = numerator / (math.sqrt(left_energy * right_energy) + 1e-8)
        acf[index] = correlation
        continuous_bin = 256.0 * spacing / period
        powers = np.asarray(
            [
                interpolate_power(power, continuous_bin),
                interpolate_power(power, 2.0 * continuous_bin),
                interpolate_power(power, 3.0 * continuous_bin),
            ],
            dtype=np.float64,
        )
        harmonic[index] = powers
        spectral = float(
            (powers[0] + 0.5 * powers[1] + 0.25 * powers[2])
            / (1.75 * spectral_denominator + 1e-8)
        )
        score[index] = 0.5 * correlation + 0.5 * spectral
        score_valid[index] = correlation >= 0.25
    local_maximum = np.zeros(candidates.shape, dtype=bool)
    for index in range(candidates.size):
        if not score_valid[index]:
            continue
        if candidates.size == 1:
            local_maximum[index] = True
        elif index == 0:
            local_maximum[index] = score[index] >= score[index + 1]
        elif index == candidates.size - 1:
            local_maximum[index] = score[index] >= score[index - 1]
        else:
            local_maximum[index] = score[index] >= score[index - 1] and score[index] >= score[index + 1]
    eligible = np.flatnonzero(local_maximum)
    selected_index = None if not eligible.size else min(
        (int(index) for index in eligible),
        key=lambda index: (-float(score[index]), int(candidates[index])),
    )
    result = {
        "expected_velocity.npy": velocity,
        "expected_velocity_mask.npy": velocity_mask.astype("|u1"),
        "expected_fft_x.npy": fft_x,
        "expected_fft_mask.npy": fft_mask.astype("|u1"),
        "expected_fft_power.npy": power[1:128],
        "expected_candidate_present.npy": np.zeros(125, dtype="|u1"),
        "expected_acf.npy": np.zeros(125, dtype="<f8"),
        "expected_acf_computed.npy": np.zeros(125, dtype="|u1"),
        "expected_harmonic_power.npy": np.zeros((125, 3), dtype="<f8"),
        "expected_score.npy": np.zeros(125, dtype="<f8"),
        "expected_score_computed.npy": np.zeros(125, dtype="|u1"),
        "expected_score_valid.npy": np.zeros(125, dtype="|u1"),
        "expected_local_max.npy": np.zeros(125, dtype="|u1"),
        "expected_selected_period.npy": np.asarray(
            (-1 if selected_index is None else candidates[selected_index],), dtype="<i2"
        ),
    }
    indices = candidates - 4
    finite_acf = np.isfinite(acf)
    finite_score = np.isfinite(score)
    result["expected_candidate_present.npy"][indices] = 1
    result["expected_acf.npy"][indices] = np.where(finite_acf, acf, 0.0)
    result["expected_acf_computed.npy"][indices] = finite_acf
    result["expected_harmonic_power.npy"][indices] = np.where(finite_score[:, None], harmonic, 0.0)
    result["expected_score.npy"][indices] = np.where(finite_score, score, 0.0)
    result["expected_score_computed.npy"][indices] = finite_score
    result["expected_score_valid.npy"][indices] = score_valid
    result["expected_local_max.npy"][indices] = local_maximum
    return result


def main() -> None:
    expected_schema = {"candidate_period.npy": ("<u2", (125,))}
    for fixture_id, (clock_count, *_rest) in SPECS.items():
        for suffix, (dtype, shape) in SUFFIX_SCHEMAS.items():
            expected_schema[f"{fixture_id}.{suffix}"] = (dtype, shape(clock_count))
    pack_members = set(expected_schema) | {"metadata.json"}
    root_members = pack_members | {"selector-unit-fixtures-v1.receipt.json"}
    assert {path.name for path in PACK.iterdir()} == root_members
    metadata_bytes = (PACK / "metadata.json").read_bytes()
    receipt_bytes = (PACK / "selector-unit-fixtures-v1.receipt.json").read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    for payload, parsed in ((metadata_bytes, metadata), (receipt_bytes, receipt)):
        assert not payload.startswith(b"\xef\xbb\xbf") and not payload.endswith((b"\n", b"\r"))
        assert payload == canonical_json_bytes(parsed)
    assert receipt["pack_member_order"] == sorted(pack_members)
    files: dict[str, bytes] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(pack_members):
        payload = (PACK / name).read_bytes()
        files[name] = payload
        assert sha256(payload) == receipt["member_sha256"][name]
        if not name.endswith(".npy"):
            continue
        handle = io.BytesIO(payload)
        assert npy_format.read_magic(handle) == (2, 0)
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
        expected_dtype, expected_shape = expected_schema[name]
        assert dtype.str == expected_dtype and shape == expected_shape and not fortran_order
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        assert array.flags.c_contiguous and not array.dtype.hasobject
        if array.dtype.kind == "f":
            assert np.isfinite(array).all()
        arrays[name] = array
    digest = hashlib.sha256()
    for name in sorted(pack_members):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[name])
        digest.update(b"\n")
    assert digest.hexdigest() == receipt["pack_sha256"]
    for key in ("amendment_sha256", "acceptance_review_sha256", "canonical_document_sha256", "source_sha256"):
        assert metadata[key] == receipt[key]
        for relative, expected_hash in receipt[key].items():
            assert sha256((ROOT / relative).read_bytes()) == expected_hash
    unit_source = (ROOT / "scripts/experiments/generate_warp_phase_selector_unit_fixtures.py").read_text(encoding="utf-8")
    random_calls = []
    for node in ast.walk(ast.parse(unit_source)):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func).lower()
            if any(token in name for token in ("random", "pcg", "rng", "secrets")):
                random_calls.append(name)
    assert not random_calls
    stochastic_source = (ROOT / "scripts/experiments/generate_warp_phase_fixture_pack.py").read_text(encoding="utf-8")
    for token in (*SPECS, "selector-unit-fixtures-v1"):
        assert token not in stochastic_source
    gate_source = (ROOT / "src/pams/warp_phase/gates.py").read_text(encoding="utf-8")
    gate_tree = ast.parse(gate_source)
    for function_name in ("_gate2_fixture_statistics", "verify_gate2_fixture_thresholds", "gate2_selector_receipt"):
        node = next(
            item
            for item in ast.walk(gate_tree)
            if isinstance(item, ast.FunctionDef) and item.name == function_name
        )
        segment = ast.get_source_segment(gate_source, node)
        for token in (*SPECS, "selector-unit-fixtures-v1", "verify_selector_unit_fixture_pack"):
            assert token not in segment
    errors: dict[str, float] = {}
    findings: dict[str, object] = {}

    def compare(name: str, observed: np.ndarray, expected: np.ndarray) -> None:
        if observed.dtype.kind == "f":
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12, equal_nan=False)
            errors[name] = float(np.max(np.abs(observed - expected))) if observed.size else 0.0
        else:
            np.testing.assert_array_equal(observed, expected.astype(observed.dtype))
            errors[name] = 0.0

    for fixture_id, (clock_count, source_period, span, p_max, selected_period) in SPECS.items():
        prefix = f"{fixture_id}."
        clocks = arrays[prefix + "input_clocks.npy"]
        pose = arrays[prefix + "input_pose.npy"]
        mask = arrays[prefix + "input_joint_mask.npy"]
        expected_clocks, expected_pose, expected_mask = constructor(clock_count, source_period)
        np.testing.assert_array_equal(clocks, expected_clocks)
        np.testing.assert_allclose(pose, expected_pose, rtol=1e-12, atol=1e-12, equal_nan=False)
        np.testing.assert_array_equal(mask, expected_mask)
        scale, coordinates, joint_valid, frame_valid = normalize(pose, mask)
        compare(prefix + "expected_scale.npy", arrays[prefix + "expected_scale.npy"], np.asarray((scale,), dtype="<f8"))
        compare(prefix + "expected_normalized_coordinates.npy", arrays[prefix + "expected_normalized_coordinates.npy"], coordinates)
        compare(prefix + "expected_normalized_joint_mask.npy", arrays[prefix + "expected_normalized_joint_mask.npy"], joint_valid)
        compare(prefix + "expected_feature_frame_valid.npy", arrays[prefix + "expected_feature_frame_valid.npy"], frame_valid)
        recomputed = recompute_selector(clocks, coordinates, joint_valid)
        for suffix, expected in recomputed.items():
            compare(prefix + suffix, arrays[prefix + suffix], expected)
        assert scale == 2.0 and joint_valid.all() and frame_valid.all()
        assert recomputed["expected_velocity_mask.npy"].all() and recomputed["expected_fft_mask.npy"].all()
        assert int(recomputed["expected_selected_period.npy"][0]) == selected_period
        acf_computed = arrays[prefix + "expected_acf_computed.npy"].astype(bool)
        score_computed = arrays[prefix + "expected_score_computed.npy"].astype(bool)
        acf = arrays[prefix + "expected_acf.npy"]
        score = arrays[prefix + "expected_score.npy"]
        harmonic = arrays[prefix + "expected_harmonic_power.npy"]
        for suffix in SUFFIX_SCHEMAS:
            if suffix != "expected_selected_period.npy":
                assert not np.any(arrays[prefix + suffix] == -1)
        assert np.all(acf[~acf_computed] == 0.0) and not np.signbit(acf[~acf_computed]).any()
        assert np.all(score[~score_computed] == 0.0) and not np.signbit(score[~score_computed]).any()
        assert np.all(harmonic[~score_computed] == 0.0) and not np.signbit(harmonic[~score_computed]).any()
        fft_x = arrays[prefix + "expected_fft_x.npy"]
        delta = float((fft_x[-1] - fft_x[0]) / np.float64(255.0))
        local = arrays[prefix + "expected_local_max.npy"]
        row: dict[str, object] = {
            "acf_computed": int(acf_computed.sum()),
            "delta": delta,
            "local_max_count": int(local.sum()),
            "p_max": p_max,
            "score_computed": int(score_computed.sum()),
            "score_valid": int(arrays[prefix + "expected_score_valid.npy"].sum()),
            "selected_period": selected_period,
            "span": span,
        }
        if fixture_id == "offbin_delta1_p20":
            assert fft_x.tobytes() == np.arange(256, dtype="<f8").tobytes()
            continuous_bin = np.float64(256.0) * np.float64(delta) / np.float64(20.0)
            power = arrays[prefix + "expected_fft_power.npy"]
            interpolation = (
                (np.float64(13.0) - continuous_bin) * power[11]
                + (continuous_bin - np.float64(12.0)) * power[12]
            )
            assert float(continuous_bin) == 12.8 and harmonic[16, 0] == interpolation
            row.update(
                bins=[12, 13],
                k=12.8,
                score_p19=float(score[15]),
                score_p20=float(score[16]),
                score_p21=float(score[17]),
            )
        elif fixture_id == "lower_endpoint_p4":
            assert score[0] >= score[1] and local[0]
            row.update(score_p4=float(score[0]), score_p5=float(score[1]))
        else:
            assert score[59] >= score[58] and local[59]
            row.update(score_p62=float(score[58]), score_p63=float(score[59]))
        findings[fixture_id] = row
    assert metadata["decision_failure_sentinel"] == -1
    assert metadata["diagnostic_fill"] == 0.0
    assert all(
        value == {"failure_reason": None, "stage": "complete"}
        for value in metadata["diagnostic_status"].values()
    )
    result = {
        "atol": 1e-12,
        "canonical_json": True,
        "equal_nan": False,
        "fixtures": findings,
        "gate2_separate": True,
        "live_bindings": True,
        "max_abs_error_all_float_recomputations": max(errors.values()),
        "metadata_sha256": sha256(metadata_bytes),
        "no_rng": True,
        "npy_member_count": len(arrays),
        "npy_v2_schema_c_order_finite": True,
        "numpy": np.__version__,
        "pack_member_count": len(pack_members),
        "pack_sha256": digest.hexdigest(),
        "python": sys.version.split()[0],
        "receipt_sha256": sha256(receipt_bytes),
        "root_member_count": len(root_members),
        "rtol": 1e-12,
        "stochastic_pack_separate": True,
    }
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")


if __name__ == "__main__":
    main()
