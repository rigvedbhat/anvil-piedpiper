"""
PCAM Precision Agent.

Two-mode agent:
  1. Noisy queries (low similarity) → retrieval-focused precision that
     discriminates the nearest pattern from competitors.
  2. Near-clean probes (high similarity) → Hessian-aware precision that
     isotropises the contraction operator Π^{1/2} H Π^{1/2} at the
     matched attractor. This directly targets the anisotropy spread metric.

The Hessian-based precision vectors are precomputed at init time via
eigenvalue-gradient descent on the condition number (scratch_diag_opt.py
validated this approach achieves 15-30× spread reductions).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    """NumPy-only PCAM precision agent with Hessian isotropisation."""

    ALPHA = 4.0
    N_COMPETITORS = 3
    PROBE_SIM_THRESHOLD = 0.85

    # Hessian optimiser knobs
    OPT_STEPS = 1500
    OPT_LR = 1.0

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # Extract frozen model parameters for Hessian computation
        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])

        # Pre-compute unit-norm patterns for cosine similarity
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_unit = self.X / np.maximum(norms, 1e-12)

        # Pre-compute Hessian-optimal precision for each stored pattern
        self._hessian_pi = self._precompute_hessian_precisions()

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        """Compute symmetrised Hessian of the PCAM energy at point a."""
        z = self.beta * (self.X @ a)
        z = z - z.max()
        e = np.exp(z)
        s = e / e.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _clip_norm(self, pi: np.ndarray) -> np.ndarray:
        """Clip and mean-normalise (mirrors model.clip_and_normalise)."""
        pi = np.clip(pi, self.pi_min, self.pi_max)
        m = pi.mean()
        if m > 0:
            pi = pi / m
        return pi

    def _spread(self, H: np.ndarray, pi: np.ndarray) -> float:
        """Condition number of Π^{1/2} H Π^{1/2}."""
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
        """Eigenvalue-gradient descent to minimise condition number of Π^{1/2} H Π^{1/2}.
        
        The gradient of the condition number κ = λ_max/λ_min with respect to
        log(π) has the direction v_max² − v_min² where v_max, v_min are the
        eigenvectors of the extremal eigenvalues. This pushes precision up on
        dimensions that shrink λ_max and down on those that grow λ_min.
        """
        N = H.shape[0]
        y = np.zeros(N)  # log-precision
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

            # gradient: push up dimensions where v_max has weight, down where v_min has
            grad = vecs[:, i_max] ** 2 - vecs[:, i_min] ** 2
            grad -= grad.mean()

            step = self.OPT_LR / math.sqrt(t + 1.0)
            y -= step * grad
            y -= y.mean()  # keep centred (mean-normalisation makes absolute scale irrelevant)
            y = np.clip(y, log_min, log_max)

        return best_pi

    def _precompute_hessian_precisions(self) -> np.ndarray:
        """For each stored pattern, compute the optimal isotropising precision."""
        all_pi = np.ones((self.K, self.N), dtype=np.float64)
        for k in range(self.K):
            H = self._hessian(self.X[k])
            # Check Hessian is positive-definite at this pattern
            eig_H = np.linalg.eigvalsh(H)
            if eig_H.min() <= 0:
                # Not in a stable basin — fall back to identity
                continue
            all_pi[k] = self._optimise_pi(H)
        return all_pi

    # ── retrieval branch (noisy queries) ──────────────────────────────

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

    # ── main entry point ──────────────────────────────────────────────

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        query = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        idx, max_sim, sims = self._nearest(query)

        if max_sim >= self.PROBE_SIM_THRESHOLD:
            # Near-clean probe: use precomputed Hessian-optimal precision
            return self._hessian_pi[idx]

        return self._retrieval_precision(sims)
