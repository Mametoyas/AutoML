"""
Particle Swarm Optimization (PSO) with linearly decaying inertia weight.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class ParticleSwarmOptimization(BaseOptimizer):
    """
    PSO with linearly decaying inertia weight w: 0.9 → 0.4
    """

    optimizer_name = "Particle Swarm Optimization (PSO)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 c1=1.5, c2=1.5,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.c1 = c1
        self.c2 = c2
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def optimize(self):
        """Run PSO optimization."""
        self._start_timer()

        # Initialize positions and velocities
        positions = np.array([self.random_vector() for _ in range(self.pop_size)])
        velocities = np.zeros((self.pop_size, self.dim))

        # Evaluate initial positions
        fitness_values = np.array([self._evaluate(p) for p in positions])

        pbest = positions.copy()
        pbest_fitness = fitness_values.copy()

        best_idx = np.argmax(pbest_fitness)
        gbest = pbest[best_idx].copy()
        gbest_fitness = pbest_fitness[best_idx]

        self.best_vector = gbest.copy()
        self.best_fitness = gbest_fitness

        for t in range(self.max_iter):
            w = 0.9 - 0.5 * (t / self.max_iter)  # linear decay 0.9 → 0.4

            for i in range(self.pop_size):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                velocities[i] = (w * velocities[i] +
                                  self.c1 * r1 * (pbest[i] - positions[i]) +
                                  self.c2 * r2 * (gbest - positions[i]))

                positions[i] = self.clip_vector(positions[i] + velocities[i])

                fit = self._evaluate(positions[i])
                fitness_values[i] = fit

                if fit > pbest_fitness[i]:
                    pbest[i] = positions[i].copy()
                    pbest_fitness[i] = fit

                    if fit > gbest_fitness:
                        gbest = positions[i].copy()
                        gbest_fitness = fit
                        self.best_vector = gbest.copy()
                        self.best_fitness = gbest_fitness

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[PSO] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
