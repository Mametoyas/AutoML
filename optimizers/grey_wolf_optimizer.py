"""
Grey Wolf Optimizer (GWO) - REQUIRED by specification.
Mirjalili et al. (2014). Advances in Engineering Software.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer with alpha-beta-delta hierarchy.
    REQUIRED by specification - appears in all default runs.
    """

    optimizer_name = "Grey Wolf Optimizer (GWO)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def optimize(self):
        """Run GWO optimization."""
        self._start_timer()

        # Initialize wolf pack
        wolves = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(w) for w in wolves])

        # Sort and identify alpha, beta, delta
        sorted_idx = np.argsort(fitness_values)[::-1]
        alpha = wolves[sorted_idx[0]].copy()
        alpha_fit = fitness_values[sorted_idx[0]]
        beta = wolves[sorted_idx[1]].copy() if self.pop_size > 1 else alpha.copy()
        beta_fit = fitness_values[sorted_idx[1]] if self.pop_size > 1 else alpha_fit
        delta = wolves[sorted_idx[2]].copy() if self.pop_size > 2 else beta.copy()
        delta_fit = fitness_values[sorted_idx[2]] if self.pop_size > 2 else beta_fit

        self.best_vector = alpha.copy()
        self.best_fitness = alpha_fit

        for t in range(self.max_iter):
            a = 2.0 - 2.0 * (t / self.max_iter)  # linearly decreases 2 → 0

            for i in range(self.pop_size):
                new_pos = np.zeros(self.dim)

                for d in range(self.dim):
                    # Alpha guides
                    r1, r2 = self.rng.random(), self.rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - wolves[i][d])
                    X1 = alpha[d] - A1 * D_alpha

                    # Beta guides
                    r1, r2 = self.rng.random(), self.rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - wolves[i][d])
                    X2 = beta[d] - A2 * D_beta

                    # Delta guides
                    r1, r2 = self.rng.random(), self.rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - wolves[i][d])
                    X3 = delta[d] - A3 * D_delta

                    new_pos[d] = (X1 + X2 + X3) / 3.0

                new_pos = self.clip_vector(new_pos)
                new_fit = self._evaluate(new_pos)

                if new_fit > fitness_values[i]:
                    wolves[i] = new_pos
                    fitness_values[i] = new_fit

            # Re-identify alpha, beta, delta
            sorted_idx = np.argsort(fitness_values)[::-1]
            alpha = wolves[sorted_idx[0]].copy()
            alpha_fit = fitness_values[sorted_idx[0]]
            beta = wolves[sorted_idx[1]].copy() if self.pop_size > 1 else alpha.copy()
            delta = wolves[sorted_idx[2]].copy() if self.pop_size > 2 else beta.copy()

            if alpha_fit > self.best_fitness:
                self.best_fitness = alpha_fit
                self.best_vector = alpha.copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[GWO] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
