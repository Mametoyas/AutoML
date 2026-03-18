"""
Cuckoo Search (CS) - Bonus optimizer.
Yang & Deb (2009). NaBIC 2009.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class CuckooSearch(BaseOptimizer):
    """Cuckoo Search with Levy flights and nest abandonment."""

    optimizer_name = "Cuckoo Search (CS)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 pa=0.25, alpha=0.01,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.pa = pa
        self.alpha = alpha
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def _levy_flight(self, dim):
        """Levy flight using Mantegna's algorithm, beta=1.5."""
        beta = 1.5
        sigma = ((0.8862 * np.sin(np.pi * 0.75)) /
                 (0.9064 * 1.5 * 2 ** 0.25)) ** (1 / 1.5)
        u = self.rng.normal(0, sigma, dim)
        v = self.rng.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / 1.5))
        return step

    def optimize(self):
        """Run Cuckoo Search optimization."""
        self._start_timer()

        nests = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(n) for n in nests])

        best_idx = np.argmax(fitness_values)
        best_nest = nests[best_idx].copy()
        best_fit = fitness_values[best_idx]
        self.best_vector = best_nest.copy()
        self.best_fitness = best_fit

        for t in range(self.max_iter):
            # Levy flight update
            for i in range(self.pop_size):
                step = self._levy_flight(self.dim)
                new_nest = nests[i] + self.alpha * step * (nests[i] - best_nest)
                new_nest = self.clip_vector(new_nest)

                # Replace random nest j (j != i)
                candidates = [j for j in range(self.pop_size) if j != i]
                j = self.rng.choice(candidates)

                new_fit = self._evaluate(new_nest)
                if new_fit > fitness_values[j]:
                    nests[j] = new_nest
                    fitness_values[j] = new_fit

            # Nest abandonment
            n_abandon = int(self.pa * self.pop_size)
            sort_idx = np.argsort(fitness_values)  # ascending
            for idx in sort_idx[:n_abandon]:
                nests[idx] = self.random_vector()
                fitness_values[idx] = self._evaluate(nests[idx])

            # Update best
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_vector = nests[best_idx].copy()
                best_nest = self.best_vector.copy()
                best_fit = self.best_fitness

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[CS]  Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
