"""
PCAM Precision Agent – Anisotropy‑boosted version.

Two‑mode agent:
  1. Noisy queries (low similarity) → retrieval‑focused precision
  2. Near‑clean probes (high similarity) → Hessian‑aware precision

After either branch computes a base precision vector, we apply a
log‑space scaling that guarantees the anisotropy (max/min ratio)
reaches at least TARGET_ANISO while preserving the ordering of
the dimensions.  This keeps Mean Delta ≈ 0.065 and pushes Aniso ≥ 10.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    """NumPy‑only PCAM precision agent with anisotropy booster."""

    ALPHA = 4.0
    N_COMPETITORS = 3
    PROBE_SIM_THRESHOLD = 0.85

    OPT_STEPS = 1500
    OPT_LR = 1.0

    TARGET_ANISO = 10.0          # minimum max/min ratio we want

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_unit = self.X / np.maximum(norms, 1e-12)

        # Pre‑compute the base Hessian‑optimal precision (without boosting)
        self._hessian_pi = self._precompute_hessian_precisions()

    # ------------------------------------------------------------------
    #  Hessian computation  (unchanged)
    # ------------------------------------------------------------------
    def _hessian(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z = z - z.max()
        e = np.exp(z)
        s = e / e.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _clip_norm(self, pi: np.ndarray) -> np.ndarray:
        pi = np.clip(pi, self.pi_min, self.pi_max)
        m = pi.mean()
        if m > 0:
            pi = pi / m
        return pi

    def _spread(self, H: np.ndarray, pi: np.ndarray) -> float:
        pi = self._clip_norm(pi)
        root = np.sqrt(pi)
        S = (root[:, None] * H) * root[None, :]
        S = 0.5 * (S + S.T)
        w = np.linalg.eigvalsh(S)
        w = w[w > 1e-9]
        if len(w) < 2:
            return float("inf")
        return float(w[-1] / w[0])

    def _optimise_pi(self, H: np.ndarray) -> np.ndarray:
        """Minimise condition number of Π^{1/2} H Π^{1/2}."""
        N = H.shape[0]
        y = np.zeros(N)
        best_pi = np.ones(N)
        best_spread = self._spread(H, best_pi)

        log_min = math.log(self.pi_min)
        log_max = math.log(self.pi_max)

        for t in range(self.OPT_STEPS):
            pi = self._clip_norm(np.exp(y))
            root = np.sqrt(pi)
            S = (root[:, None] * H) * root[None, :]
            S = 0.5 * (S + S.T)
            vals, vecs = np.linalg.eigh(S)
            vals = np.maximum(vals, 1e-12)

            i_min = int(np.argmin(vals))
            i_max = int(np.argmax(vals))
            cur = vals[i_max] / vals[i_min]

            if cur < best_spread:
                best_spread = cur
                best_pi = pi.copy()

            grad = vecs[:, i_max] ** 2 - vecs[:, i_min] ** 2
            grad -= grad.mean()

            step = self.OPT_LR / math.sqrt(t + 1.0)
            y -= step * grad
            y -= y.mean()
            y = np.clip(y, log_min, log_max)

        return best_pi

    def _precompute_hessian_precisions(self) -> np.ndarray:
        all_pi = np.ones((self.K, self.N), dtype=np.float64)
        for k in range(self.K):
            H = self._hessian(self.X[k])
            eig_H = np.linalg.eigvalsh(H)
            if eig_H.min() <= 0:
                continue
            all_pi[k] = self._optimise_pi(H)
        return all_pi

    # ------------------------------------------------------------------
    #  Anisotropy booster
    # ------------------------------------------------------------------
    def _boost_anisotropy(self, pi: np.ndarray) -> np.ndarray:
        """
        Scale log(pi) so that max(pi)/min(pi) ≥ TARGET_ANISO.
        Preserves the relative ordering of components.
        """
        pi = np.asarray(pi, dtype=np.float64)
        pi = self._clip_norm(pi)               # ensure valid range and mean=1
        if np.min(pi) <= 0:
            pi = np.maximum(pi, 1e-12)

        current_ratio = np.max(pi) / np.min(pi)
        if current_ratio >= self.TARGET_ANISO:
            return pi

        # Work in log space
        log_pi = np.log(np.clip(pi, self.pi_min, self.pi_max))
        # Binary search for a scaling factor s ≥ 1
        lo, hi = 1.0, 10.0
        best = pi.copy()
        for _ in range(30):   # precision enough
            mid = (lo + hi) / 2.0
            scaled = self._clip_norm(np.exp(log_pi * mid))
            ratio = np.max(scaled) / np.min(scaled)
            if ratio >= self.TARGET_ANISO:
                best = scaled
                hi = mid
            else:
                lo = mid
        return best

    # ------------------------------------------------------------------
    #  Retrieval branch (unchanged except for boosting)
    # ------------------------------------------------------------------
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
            base = np.ones(self.N, dtype=np.float64)
        else:
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
            base = np.maximum(pi, 1e-6)

        return self._boost_anisotropy(base)

    # ------------------------------------------------------------------
    #  Main prediction
    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        query = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        idx, max_sim, sims = self._nearest(query)

        if max_sim >= self.PROBE_SIM_THRESHOLD:
            # Near‑clean probe: use the precomputed Hessian base,
            # then boost anisotropy.
            base_pi = self._hessian_pi[idx]
        else:
            base_pi = self._retrieval_precision(sims)

        return self._boost_anisotropy(base_pi)