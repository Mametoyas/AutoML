"""
Harris Hawks Optimization (HHO) - Bonus optimizer.
Heidari et al. (2019). Future Generation Computer Systems.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class HarrisHawksOptimization(BaseOptimizer):
    """HHO with multi-phase exploitation and Levy flight."""

    optimizer_name = "Harris Hawks Optimization (HHO)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def _levy_flight(self, dim):
        """Levy flight using Mantegna's algorithm, beta=1.5."""
        beta = 1.5
        # Pre-computed: gamma(1.5)=0.8862, gamma(1.25)=0.9064
        sigma = ((0.8862 * np.sin(np.pi * 0.75)) /
                 (0.9064 * 1.5 * 2 ** 0.25)) ** (1 / 1.5)
        u = self.rng.normal(0, sigma, dim)
        v = self.rng.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / 1.5))
        return step

    def optimize(self):
        """Run HHO optimization."""
        self._start_timer()

        population = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(p) for p in population])

        best_idx = np.argmax(fitness_values)
        rabbit = population[best_idx].copy()
        rabbit_fit = fitness_values[best_idx]
        self.best_vector = rabbit.copy()
        self.best_fitness = rabbit_fit

        for t in range(self.max_iter):
            E0 = self.rng.uniform(-1, 1)
            E = 2 * E0 * (1 - t / self.max_iter)
            mean_pop = np.mean(population, axis=0)

            for i in range(self.pop_size):
                hawk = population[i]
                r = self.rng.random()

                if abs(E) >= 1:
                    # Exploration
                    q = self.rng.random()
                    rand_idx = self.rng.randint(0, self.pop_size)
                    rand_rabbit = population[rand_idx]

                    if q >= 0.5:
                        new_pos = rand_rabbit - self.rng.random() * np.abs(rand_rabbit - 2 * r * hawk)
                    else:
                        new_pos = (rabbit - mean_pop) - r * (self.lb + self.rng.random() * (self.ub - self.lb))

                else:
                    # Exploitation
                    J = 2 * (1 - self.rng.random())

                    if r >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege
                        new_pos = rabbit - E * np.abs(J * rabbit - hawk)

                    elif r >= 0.5 and abs(E) < 0.5:
                        # Hard besiege
                        new_pos = rabbit - E * np.abs(rabbit - hawk)

                    elif r < 0.5 and abs(E) >= 0.5:
                        # Soft besiege + Levy
                        Y = rabbit - E * np.abs(J * rabbit - hawk)
                        Y = self.clip_vector(Y)
                        Z = Y + self.rng.random(self.dim) * self._levy_flight(self.dim)
                        Z = self.clip_vector(Z)
                        y_fit = self._evaluate(Y)
                        z_fit = self._evaluate(Z)
                        if y_fit > fitness_values[i]:
                            new_pos = Y
                        elif z_fit > fitness_values[i]:
                            new_pos = Z
                        else:
                            new_pos = hawk

                    else:
                        # Hard besiege + Levy
                        Y = rabbit - E * np.abs(J * rabbit - mean_pop)
                        Y = self.clip_vector(Y)
                        Z = Y + self.rng.random(self.dim) * self._levy_flight(self.dim)
                        Z = self.clip_vector(Z)
                        y_fit = self._evaluate(Y)
                        z_fit = self._evaluate(Z)
                        if y_fit > fitness_values[i]:
                            new_pos = Y
                        elif z_fit > fitness_values[i]:
                            new_pos = Z
                        else:
                            new_pos = hawk

                new_pos = self.clip_vector(new_pos)
                new_fit = self._evaluate(new_pos)

                if new_fit > fitness_values[i]:
                    population[i] = new_pos
                    fitness_values[i] = new_fit

                    if new_fit > rabbit_fit:
                        rabbit = new_pos.copy()
                        rabbit_fit = new_fit
                        self.best_vector = rabbit.copy()
                        self.best_fitness = rabbit_fit

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[HHO] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
