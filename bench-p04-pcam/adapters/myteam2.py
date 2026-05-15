"""
PCAM Precision Agent – Bench v2 Clean Implementation.

Refactored for the updated PCAM benchmark.
Key changes:
1. Iterative projection for normalization (satisfies pi_min, pi_max, and mean=1).
2. Computes proper equilibrium a* for Hessian geometry.
3. Fast intelligent precomputation of geometry.
4. Improved cluster-aware retrieval.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    """Bench v2 Generalized Precision Agent."""

    ALPHA = 6.0
    N_COMPETITORS = 3
    
    OPT_STEPS = 1500
    OPT_LR = 2.0

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # Copy to avoid any accidental benchmark mutation
        self.R = np.asarray(model_params["R"], dtype=np.float64).copy()
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_unit = self.X / np.maximum(norms, 1e-12)

        # Pre-compute the optimal geometry-aware precision using true equilibria
        self._hessian_pi = self._precompute_hessian_precisions()

    # ------------------------------------------------------------------
    #  Normalization
    # ------------------------------------------------------------------
    def _clip_and_normalise(self, pi: np.ndarray) -> np.ndarray:
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        if not np.all(np.isfinite(pi)):
            return np.ones(self.N)

        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = pi.mean()
            if m <= 1e-12:
                return np.ones(self.N)
            pi = pi / m
            within_bounds = (pi.min() >= self.pi_min - 1e-9
                             and pi.max() <= self.pi_max + 1e-9)
            mean_ok = abs(pi.mean() - 1.0) < 1e-8
            if within_bounds and mean_ok:
                break

        return np.clip(pi, self.pi_min, self.pi_max)

    # ------------------------------------------------------------------
    #  Hessian computation around TRUE equilibrium a*
    # ------------------------------------------------------------------
    def _softmax(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _gradient(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        return self.R @ a - self.eta * (self.X.T @ s)

    def _find_equilibrium(self, x0: np.ndarray) -> np.ndarray:
        """Run dynamics from x0 with pi = I to find true a*."""
        a = x0.copy()
        dt = 0.01
        tol = 1e-6
        for _ in range(3000):
            g = self._gradient(a)
            a_new = a - dt * g
            if np.linalg.norm(a_new - a) < tol:
                a = a_new
                break
            a = a_new
        return a

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)
        
    def _spread(self, H: np.ndarray, pi: np.ndarray) -> float:
        pi = self._clip_and_normalise(pi)
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
            pi = self._clip_and_normalise(np.exp(y))
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
            # Compute true equilibrium a* to match the benchmark's evaluation point
            a_star = self._find_equilibrium(self.X[k])
            H = self._hessian(a_star)
            eig_H = np.linalg.eigvalsh(H)
            if eig_H.min() <= 0:
                continue
            all_pi[k] = self._optimise_pi(H)
        return all_pi

    # ------------------------------------------------------------------
    #  Retrieval branch
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
            return np.ones(self.N, dtype=np.float64)

        score = np.zeros(self.N, dtype=np.float64)
        weight_sum = 0.0
        
        for idx in competitors:
            diff = np.abs(target - self.X[idx])
            
            # Density weighting: give more importance to closer, more confusable competitors
            weight = math.exp(4.0 * sims[idx])
            
            # Scale the difference
            scale = float(np.max(diff))
            if scale > 1e-12:
                diff = diff / scale
                
            score += weight * diff
            weight_sum += weight

        if weight_sum > 1e-12:
            score /= weight_sum
            
        scale = float(np.max(score))
        if scale > 1e-12:
            score = score / scale
            
        score = score - float(np.mean(score))

        pi = np.exp(self.ALPHA * score)
        return self._clip_and_normalise(pi)

    # ------------------------------------------------------------------
    #  Main prediction (Hard Threshold)
    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        query = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        idx, max_sim, sims = self._nearest(query)

        # For Bench v2, use hard threshold. 
        # Clean probes (high sim) use pure geometry (tests anisotropy).
        # Noisy queries (low sim) use pure retrieval.
        if max_sim >= 0.80:
            return self._hessian_pi[idx]
        else:
            return self._retrieval_precision(sims)
