"""
Differential Evolution (DE) optimizer - DE/rand/1/bin strategy.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution with DE/rand/1/bin strategy.
    """

    optimizer_name = "Differential Evolution (DE)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 F=0.8, CR=0.9,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.F = F
        self.CR = CR
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def optimize(self):
        """Run DE optimization."""
        self._start_timer()

        # Initialize
        population = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(p) for p in population])

        best_idx = np.argmax(fitness_values)
        self.best_vector = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Select r1, r2, r3 distinct and != i
                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                mutant = self.clip_vector(mutant)

                # Binomial crossover
                j_rand = self.rng.randint(0, self.dim)
                trial = np.where(
                    (self.rng.random(self.dim) < self.CR) | (np.arange(self.dim) == j_rand),
                    mutant,
                    population[i]
                )

                # Greedy selection
                trial_fitness = self._evaluate(trial)
                if trial_fitness >= fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

                    if trial_fitness > self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_vector = trial.copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[DE]  Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
