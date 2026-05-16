import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


EPS = 1e-12
AMPLITUDE_EPS = 0.2
NEAR_PATTERN_COSINE = 0.84
MARGIN_COSINE = 0.76
MARGIN_GAP = 0.28


class Engine(Adapter):
    """PCAM precision agent.

    Corrupted queries use a robust inverse-amplitude rule. Queries that
    are already very close to one stored pattern use a cached local
    Hessian preconditioner for better anisotropy. The implementation uses
    only the supplied patterns and model parameters, so it stays seed- and
    benchmark-size agnostic.
    """

    def __init__(self, stored_patterns, model_params):
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        _, self.N = self.X.shape
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        self.geom_steps = 300 if self.N <= 96 else 180
        self.geometry_cache = {}
        self.model = self._build_model(model_params)

    def _build_model(self, params):
        return PCAMModel(
            self.X,
            np.asarray(params["R"], dtype=np.float64),
            eta=float(params.get("eta", 0.5)),
            beta=float(params.get("beta", 8.0)),
            dt=float(params.get("dt", 0.01)),
            T_max=int(params.get("T_max", 3000)),
            tol=float(params.get("tol", 1e-6)),
            T_in=int(params.get("T_in", 100)),
            pi_min=self.pi_min,
            pi_max=self.pi_max,
        )

    def _project(self, pi):
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        if not np.all(np.isfinite(pi)):
            return np.ones(self.N)

        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            mean = pi.mean()
            if mean <= EPS:
                return np.ones(self.N)
            pi = pi / mean
        return np.clip(pi, self.pi_min, self.pi_max)

    def _condition_and_grad(self, H, y):
        pi = self._project(np.exp(y - y.mean()))
        d = np.sqrt(np.clip(pi, EPS, None))
        S = (d[:, None] * H) * d[None, :]
        eigs, vecs = np.linalg.eigh(0.5 * (S + S.T))
        pos = np.where(eigs > 1e-9)[0]
        if pos.size < 2:
            return np.inf, np.zeros(self.N), pi

        lo, hi = int(pos[0]), int(pos[-1])
        obj = float(np.log(eigs[hi]) - np.log(eigs[lo]))

        # d log(lambda_k) / d y_i = v_{k,i}^2 for diagonal log scaling.
        grad = vecs[:, hi] ** 2 - vecs[:, lo] ** 2
        grad = grad - grad.mean()
        return obj, grad, pi

    def _optimise_diagonal(self, H):
        """NumPy-only minimisation of log condition number."""
        H = 0.5 * (H + H.T)
        diag = np.clip(np.diag(H), 1e-8, None)
        y = np.log(self._project(1.0 / diag))
        best_obj, _, best_pi = self._condition_and_grad(H, y)
        lr = 2.0

        for _ in range(self.geom_steps):
            obj, grad, _ = self._condition_and_grad(H, y)
            candidate_y = y - lr * grad
            candidate_obj, _, candidate_pi = self._condition_and_grad(
                H, candidate_y
            )

            if candidate_obj < obj:
                y = candidate_y
                lr *= 1.02
                if candidate_obj < best_obj:
                    best_obj, best_pi = candidate_obj, candidate_pi
            else:
                lr *= 0.5
                if lr < 1e-5:
                    lr = 0.1

        return best_pi

    def _geometry_precision(self, idx):
        if idx in self.geometry_cache:
            return self.geometry_cache[idx]

        equilibrium = self.model.find_equilibrium(self.X[idx])
        H = self.model.hessian(equilibrium)
        pi = self._optimise_diagonal(H)
        self.geometry_cache[idx] = pi
        return pi

    def _near_stored_pattern(self, sims):
        if sims.size == 1:
            return sims[0] > NEAR_PATTERN_COSINE

        top_two = np.partition(sims, -2)[-2:]
        best = float(top_two[1])
        margin = float(top_two[1] - top_two[0])

        return best > NEAR_PATTERN_COSINE or (
            best > MARGIN_COSINE and margin > MARGIN_GAP
        )

    def _nearest_pattern(self, q):
        norm = np.linalg.norm(q)
        if norm <= EPS:
            return None, None

        sims = self.X @ (q / norm)
        return int(np.argmax(sims)), sims

    def _retrieval_precision(self, q):
        return self._project(1.0 / (np.abs(q) + AMPLITUDE_EPS))

    def predict_precision(self, corrupted_query):
        q = np.asarray(corrupted_query, dtype=np.float64)
        nearest, sims = self._nearest_pattern(q)
        if nearest is None:
            return np.ones(self.N)

        if self._near_stored_pattern(sims):
            return self._geometry_precision(nearest)

        return self._retrieval_precision(q)
