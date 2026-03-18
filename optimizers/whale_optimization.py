"""
Whale Optimization Algorithm (WOA) - Bonus optimizer.
Mirjalili & Lewis (2016). Advances in Engineering Software.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """WOA with bubble-net attack and random search strategies."""

    optimizer_name = "Whale Optimization Algorithm (WOA)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 b=1.0,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.b = b
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def optimize(self):
        """Run WOA optimization."""
        self._start_timer()

        population = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(p) for p in population])

        best_idx = np.argmax(fitness_values)
        best = population[best_idx].copy()
        best_fit = fitness_values[best_idx]
        self.best_vector = best.copy()
        self.best_fitness = best_fit

        for t in range(self.max_iter):
            a = 2.0 - 2.0 * (t / self.max_iter)

            for i in range(self.pop_size):
                p = self.rng.random()
                r = self.rng.random()
                A = 2 * a * r - a
                C = 2 * r

                if p < 0.5:
                    if abs(A) < 1:
                        # Exploitation: encircle prey
                        D = np.abs(C * best - population[i])
                        new_pos = best - A * D
                    else:
                        # Exploration: random whale
                        rand_idx = self.rng.randint(0, self.pop_size)
                        rand_whale = population[rand_idx]
                        D = np.abs(C * rand_whale - population[i])
                        new_pos = rand_whale - A * D
                else:
                    # Spiral bubble-net
                    l = self.rng.uniform(-1, 1)
                    D = np.abs(best - population[i])
                    new_pos = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best

                new_pos = self.clip_vector(new_pos)
                new_fit = self._evaluate(new_pos)

                if new_fit > fitness_values[i]:
                    population[i] = new_pos
                    fitness_values[i] = new_fit

                    if new_fit > best_fit:
                        best = new_pos.copy()
                        best_fit = new_fit
                        self.best_vector = best.copy()
                        self.best_fitness = best_fit

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[WOA] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
