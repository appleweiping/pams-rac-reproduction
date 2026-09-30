"""Label-free periodic reconstruction from predeclared natural landmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def periodic_template_reconstruction(
    target: NDArray[np.float64],
    mask: NDArray[np.float64],
    geometry: NDArray[np.float64],
    landmarks: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Average traversals on a geometry-arc grid and map the template back."""

    grid = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    numerator = np.zeros((grid.size, target.shape[1]), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    traversal_arcs: list[tuple[int, int, NDArray[np.float64]]] = []
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        delta = geometry[left + 1 : right + 1] - geometry[left:right]
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
        if arc[-1] <= 0.0:
            raise ValueError("periodic traversal has zero geometry arc")
        arc /= arc[-1]
        traversal_arcs.append((left, right, arc))
        for channel in range(target.shape[1]):
            valid = mask[left : right + 1, channel] > 0
            if np.count_nonzero(valid) < 2:
                continue
            numerator[:, channel] += np.interp(
                grid,
                arc[valid],
                target[left : right + 1, channel][valid],
            )
            denominator[:, channel] += 1.0
    template = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    template[-1] = template[0] = 0.5 * (template[0] + template[-1])
    reconstruction = np.array(target, dtype=np.float64, copy=True)
    for left, right, arc in traversal_arcs:
        for channel in range(target.shape[1]):
            reconstruction[left : right + 1, channel] = np.interp(
                arc,
                grid,
                template[:, channel],
            )
    return np.asarray(reconstruction, dtype="<f8")


__all__ = ["periodic_template_reconstruction"]
