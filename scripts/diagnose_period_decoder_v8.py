"""Compare period decoders on frozen dev streams without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from pams.consensus import MultiExpertCounter


def _direct_fft_period(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid]
    if selected.size < 8 or float(np.std(selected)) < 1e-8:
        return 128.0
    time = np.arange(selected.size, dtype=np.float64)
    coefficients = np.polyfit(time, selected, deg=1)
    detrended = selected - np.polyval(coefficients, time)
    windowed = detrended * np.hanning(selected.size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(selected.size)
    allowed = (frequencies >= 1.0 / 128.0) & (frequencies <= 1.0 / 4.0)
    power[~allowed] = 0.0
    if float(power.sum()) <= 1e-12:
        return 128.0
    frequency = float(frequencies[int(np.argmax(power))])
    return float(np.clip(1.0 / frequency, 4.0, 128.0))


def _acf_peak_period(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid]
    if selected.size < 8 or float(np.std(selected)) < 1e-8:
        return 128.0
    time = np.arange(selected.size, dtype=np.float64)
    coefficients = np.polyfit(time, selected, deg=1)
    centered = selected - np.polyval(coefficients, time)
    correlation = np.correlate(centered, centered, mode="full")[selected.size - 1 :]
    overlap = np.arange(selected.size, 0, -1, dtype=np.float64)
    correlation = correlation / overlap
    correlation = correlation / max(float(correlation[0]), 1e-12)
    upper = min(128, selected.size - 1)
    band = correlation[4 : upper + 1]
    peaks, properties = find_peaks(band, prominence=0.02)
    if peaks.size == 0:
        return _direct_fft_period(values, valid)
    lags = peaks + 4
    prominences = properties["prominences"]
    # Prefer a strong early recurrence over broad low-frequency drift.
    scores = correlation[lags] + prominences - 0.0015 * lags
    return float(lags[int(np.argmax(scores))])


def _metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    predicted = np.asarray(predictions)
    truth = np.asarray(targets)
    error = np.abs(predicted - truth)
    return {
        "nmae": float(np.mean(error / truth)),
        "obo": float(np.mean(error <= 1)),
        "exact": float(np.mean(error == 0)),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args()
    prediction_payload = json.loads(args.predictions.read_text())
    target_payload = json.loads(args.targets.read_text())
    target_by_id = {
        row["video_id"]: int(row["count"]) for row in target_payload["records"]
    }
    counter = MultiExpertCounter()
    results: dict[str, list[int]] = {
        "recorded": [],
        "direct_fft_detrended": [],
        "direct_fft_reference": [],
        "direct_fft_reference_round": [],
        "acf_peak_detrended": [],
        "acf_peak_reference": [],
        "acf_peak_reference_round": [],
        "acf_peak_zero_fallback": [],
    }
    targets: list[int] = []
    periods: dict[str, list[float]] = {
        "direct_fft_detrended": [],
        "acf_peak_detrended": [],
    }
    for row in prediction_payload["records"]:
        values = np.asarray(row["period_stream"], dtype=np.float64)
        valid = np.isfinite(values) & (values != 0.0)
        targets.append(target_by_id[row["video_id"]])
        results["recorded"].append(int(row["count"]))
        for name, decoder in (
            ("direct_fft_detrended", _direct_fft_period),
            ("acf_peak_detrended", _acf_peak_period),
        ):
            period = decoder(values, valid)
            periods[name].append(period)
            result = counter.count(
                values,
                period_frames=period,
                valid_mask=valid,
                period_confidence=1.0,
            )
            results[name].append(int(result.count))
            reference_name = name.replace("_detrended", "_reference")
            results[reference_name].append(int(np.floor(valid.sum() / period)))
            results[f"{reference_name}_round"].append(
                int(np.floor(valid.sum() / period + 0.5))
            )
            if name == "acf_peak_detrended":
                results["acf_peak_zero_fallback"].append(
                    int(result.reference_count if result.count == 0 else result.count)
                )

    payload = {
        name: {
            "metrics": _metrics(counts, targets),
            "zero_predictions": int(np.sum(np.asarray(counts) == 0)),
            "period_at_128": (
                None
                if name == "recorded"
                else int(
                    np.sum(
                        np.asarray(
                            periods[
                                "acf_peak_detrended"
                                if name.startswith("acf_peak")
                                else "direct_fft_detrended"
                            ]
                        )
                        == 128.0
                    )
                )
            ),
        }
        for name, counts in results.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
