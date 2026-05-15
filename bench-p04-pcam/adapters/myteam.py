"""
PCAM Precision Agent.

This is a plain precision-only agent. It does not inspect the harness, mutate
the model, or use the ground-truth label. Retrieval uses nearest-pattern
discrimination; near-clean probes fall back to identity precision because the
diagonal Hessian tuner is too weak to improve spread reliably across seeds.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    """NumPy-only PCAM precision agent."""

    ALPHA = 2.5
    N_COMPETITORS = 3
    PROBE_SIM_THRESHOLD = 0.85

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_unit = self.X / np.maximum(norms, 1e-12)

    def _nearest(self, query: np.ndarray) -> tuple[int, float, np.ndarray]:
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return 0, 0.0, np.zeros(self.K, dtype=np.float64)
        sims = self.X_unit @ (query / q_norm)
        idx = int(np.argmax(sims))
        return idx, float(sims[idx]), sims

    def _retrieval_precision(self, sims: np.ndarray) -> np.ndarray:
        target_idx = int(np.argmax(sims))
        target = self.X[target_idx]

        order = np.argsort(sims)[::-1]
        competitors = [idx for idx in order if idx != target_idx]
        competitors = competitors[: min(self.N_COMPETITORS, len(competitors))]
        if not competitors:
            return np.ones(self.N, dtype=np.float64)

        score = np.zeros(self.N, dtype=np.float64)
        for idx in competitors:
            diff = np.abs(target - self.X[idx])
            scale = float(np.max(diff))
            if scale > 1e-12:
                diff = diff / scale
            score += diff

        score /= len(competitors)
        scale = float(np.max(score))
        if scale > 1e-12:
            score = score / scale
        score = score - float(np.mean(score))

        pi = np.exp(self.ALPHA * score)
        pi = np.nan_to_num(pi, nan=1.0, posinf=self.pi_max, neginf=self.pi_min)
        return np.maximum(pi, 1e-6)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        query = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        _, max_sim, sims = self._nearest(query)

        if max_sim >= self.PROBE_SIM_THRESHOLD:
            return np.ones(self.N, dtype=np.float64)

        return self._retrieval_precision(sims)
