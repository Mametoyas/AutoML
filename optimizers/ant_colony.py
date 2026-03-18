"""
Ant Colony Optimization for continuous domains (ACOR).
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class AntColonyOptimization(BaseOptimizer):
    """
    ACOR - Ant Colony Optimization for continuous domains.
    Uses a solution archive with Gaussian kernels.
    """

    optimizer_name = "Ant Colony Optimization (ACO)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 archive_size=10, q=0.5, xi=0.85,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

        # Discrete dimensions: 0 (scaler), 1 (feature), 3 (model)
        self.discrete_dims = {0, 1, 3}

    def _compute_weights(self):
        """Compute selection weights for archive solutions."""
        S = self.archive_size
        q = self.q
        weights = np.array([
            1.0 / (q * S * np.sqrt(2 * np.pi)) * np.exp(
                -((l) ** 2) / (2 * q ** 2 * S ** 2)
            )
            for l in range(S)
        ])
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(S) / S
        return weights

    def optimize(self):
        """Run ACO optimization."""
        self._start_timer()

        # Initialize archive
        archive = np.array([self.random_vector() for _ in range(self.archive_size)])
        archive_fitness = np.array([self._evaluate(a) for a in archive])

        # Sort archive by fitness (descending)
        sort_idx = np.argsort(archive_fitness)[::-1]
        archive = archive[sort_idx]
        archive_fitness = archive_fitness[sort_idx]

        self.best_vector = archive[0].copy()
        self.best_fitness = archive_fitness[0]

        weights = self._compute_weights()
        S = self.archive_size

        for t in range(self.max_iter):
            new_solutions = []
            new_fitness_vals = []

            for _ in range(self.pop_size):
                new_sol = np.zeros(self.dim)

                for d in range(self.dim):
                    # Select archive index by roulette
                    l = self.rng.choice(S, p=weights)

                    if d in self.discrete_dims:
                        # Discrete: use archive value + small noise
                        noise = self.rng.normal(0, 0.3)
                        new_sol[d] = archive[l][d] + noise
                    else:
                        # Continuous: Gaussian kernel
                        if S > 1:
                            std = self.xi * np.sum(np.abs(archive[:, d] - archive[l][d])) / (S - 1)
                        else:
                            std = 0.1 * (self.ub[d] - self.lb[d])

                        std = max(std, 1e-10)
                        new_sol[d] = self.rng.normal(archive[l][d], std)

                new_sol = self.clip_vector(new_sol)
                fit = self._evaluate(new_sol)
                new_solutions.append(new_sol)
                new_fitness_vals.append(fit)

            # Update archive
            all_sols = np.vstack([archive, new_solutions])
            all_fitness = np.concatenate([archive_fitness, new_fitness_vals])

            sort_idx = np.argsort(all_fitness)[::-1]
            archive = all_sols[sort_idx[:S]]
            archive_fitness = all_fitness[sort_idx[:S]]

            if archive_fitness[0] > self.best_fitness:
                self.best_fitness = archive_fitness[0]
                self.best_vector = archive[0].copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[ACO] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
